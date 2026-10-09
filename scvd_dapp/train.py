"""Training orchestrator. ``run(config)`` is the single entry point.

    load table -> split files by fold -> window every split -> resample TRAIN windows
      -> train (HF Trainer for hf_encoder / hf_decoder; native loop for textcnn / openmythos)
      -> predict val + test windows -> max-pool to files -> evaluate.summarize

Model selection (best epoch, early stopping) uses the file-level macro average
precision on the validation fold -- threshold-free, so it is not at the mercy of
an arbitrary 0.5 cut-off on a rare class. Test is touched once, at the end.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Dict

import numpy as np

from . import data as datamod
from .config import BACKEND_HF_DECODER, BACKEND_HF_ENCODER, BACKEND_OPENMYTHOS, BACKEND_TEXTCNN, TrainConfig
from .evaluate import summarize
from .metrics import macro_average_precision, prf_at, sigmoid
from .windows import HFWindower, RegexWindower, WindowSet, aggregate_max, build_windows

LOGGER = logging.getLogger("scvd_dapp")


def run(config: TrainConfig) -> Dict:
    t_start = time.time()
    config.save()
    LOGGER.info("Run '%s' | backend=%s | model=%s | input=%s@%d (overlap %.2f)", config.run_name, config.backend,
                config.model_name, config.input_mode, config.max_length, config.window_overlap)

    df, tax, meta = datamod.load_table(config.parquet_path)
    out = Path(config.output_dir)
    (out / "taxonomy.json").write_text(json.dumps(
        {"classes": list(tax.swc_ids), "names": tax.names(), "dataset_version": meta.get("version"),
         "dataset_fingerprint": meta.get("fingerprint")}, indent=2), encoding="utf-8")

    train_df, val_df, test_df = datamod.split_files(df, config.test_fold, config.val_fold,
                                                    config.max_samples, config.seed)
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        datamod.log_label_distribution(part, tax, name)
    K = tax.num_classes
    Y = {n: datamod.label_matrix(p, K) for n, p in (("train", train_df), ("val", val_df), ("test", test_df))}
    spans = {n: datamod.file_spans(p) for n, p in (("train", train_df), ("val", val_df), ("test", test_df))}
    exposed_test = datamod.exposure_flags(config.parquet_path, test_df, train_df)
    LOGGER.info("Test files with a near-duplicate in train: %d / %d", int(exposed_test.sum()), len(test_df))

    # ---- tokenizer / windower --------------------------------------------------------
    tokenizer = None
    if config.backend == BACKEND_TEXTCNN:
        windower = RegexWindower(config.max_length, config.window_overlap)
        windower.build_vocab(train_df[config.text_field].tolist(), config.vocab_size, config.min_token_freq)
        LOGGER.info("TextCNN vocabulary: %d words", len(windower.vocab))
    else:
        from .models import build_tokenizer

        tokenizer = build_tokenizer(config)  # for openmythos, model_name is its tokenizer (gpt-oss-20b)
        windower = HFWindower(tokenizer, config.max_length, config.window_overlap)
    LOGGER.info("Windows: %d content tokens + %d special, stride %d", windower.body,
                len(windower.prefix) + len(windower.suffix), windower.stride)

    ws: Dict[str, WindowSet] = {}
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        ws[name] = build_windows(part[config.text_field].tolist(), spans[name], Y[name], windower,
                                 config.input_mode, K)
        LOGGER.info("%-5s windows: %s", name, ws[name].stats)
    if config.training_unit == "file":
        return _run_file_level(config, tax, meta, tokenizer, windower, ws, Y, train_df, val_df, test_df,
                               exposed_test, t_start)
    train_idx = datamod.resample_train(ws["train"], config.neg_ratio, config.oversample,
                                       config.oversample_threshold, config.oversample_target,
                                       config.oversample_max_factor, tax, config.seed)
    window_info = {"mode": config.input_mode, "max_length": config.max_length,
                   "window_overlap": config.window_overlap, "content_tokens": windower.body,
                   "stride": windower.stride, "train_windows_used": int(len(train_idx)),
                   "splits": {n: w.stats for n, w in ws.items()}}
    if config.dry_run:
        (out / "windows_report.json").write_text(json.dumps(window_info, indent=2), encoding="utf-8")
        LOGGER.info("Dry run: wrote %s; no model loaded", out / "windows_report.json")
        return window_info

    t_train = time.time()
    extra_backend: Dict = {}
    if config.backend in (BACKEND_HF_ENCODER, BACKEND_HF_DECODER):
        predict, best_val = _train_hf(config, tax, tokenizer, windower, ws, train_idx, Y["val"])
    elif config.backend == BACKEND_TEXTCNN:
        predict, best_val = _train_textcnn(config, tax, windower, ws, train_idx, Y["val"])
    elif config.backend == BACKEND_OPENMYTHOS:
        predict, best_val, extra_backend = _train_openmythos(config, tax, tokenizer, windower, ws, train_idx,
                                                             Y["val"])
    else:
        raise ValueError(config.backend)
    train_seconds = time.time() - t_train

    val_windows = predict(ws["val"])
    _check_reproduced(config.select_metric, best_val, _val_metrics(val_windows, ws["val"], Y["val"]))
    val_logits = aggregate_max(val_windows, ws["val"].file_idx, ws["val"].n_files)
    test_logits = aggregate_max(predict(ws["test"]), ws["test"].file_idx, ws["test"].n_files)
    if config.save_logits:
        # train-file logits (all windows, not the resampled set): the MoE gate is fit on these
        train_logits = aggregate_max(predict(ws["train"]), ws["train"].file_idx, ws["train"].n_files)
        np.savez_compressed(out / "logits_train.npz", logits=train_logits, labels=Y["train"].astype(np.uint8),
                            row_id=train_df["row_id"].to_numpy(), classes=np.asarray(tax.swc_ids))
    runtime = {"train_seconds": round(train_seconds, 1), "total_seconds": round(time.time() - t_start, 1)}
    try:
        import torch

        if torch.cuda.is_available():
            runtime["gpu"] = torch.cuda.get_device_name(0)
            runtime["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except ImportError:
        pass
    extra = {"run_name": config.run_name, "model_name": config.model_name, "backend": config.backend,
             "dataset": {"version": meta.get("version"), "fingerprint": meta.get("fingerprint"),
                         "test_fold": config.test_fold, "val_fold": config.val_fold,
                         "files": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
                         "max_samples": config.max_samples},
             "input": window_info, "runtime": runtime}
    if extra_backend:
        extra["backend_info"] = extra_backend
    return summarize(test_logits, Y["test"], tax, config.output_dir, optimize=config.optimize_thresholds,
                     strategy=config.threshold_strategy, temperature_scaling=config.temperature_scaling,
                     val_logits=val_logits, val_labels=Y["val"], test_exposed=exposed_test,
                     test_row_ids=test_df["row_id"].to_numpy(), val_row_ids=val_df["row_id"].to_numpy(),
                     save_logits=config.save_logits, extra=extra)


def _run_file_level(config: TrainConfig, tax, meta, tokenizer, windower, ws, Y, train_df, val_df, test_df,
                    exposed_test, t_start: float) -> Dict:
    """v2: train on whole files (scvd_dapp.filelevel), then evaluate exactly like v1."""
    out = Path(config.output_dir)
    window_info = {"mode": config.input_mode, "training_unit": "file", "max_length": config.max_length,
                   "window_overlap": config.window_overlap, "content_tokens": windower.body,
                   "stride": windower.stride, "max_windows_train": config.max_windows_train,
                   "splits": {n: w.stats for n, w in ws.items()}}
    if config.dry_run:
        (out / "windows_report.json").write_text(json.dumps(window_info, indent=2), encoding="utf-8")
        LOGGER.info("Dry run: wrote %s; no model loaded", out / "windows_report.json")
        return window_info
    from .filelevel import train_file_level

    t_train = time.time()
    res = train_file_level(config, tax, tokenizer, windower, ws, Y)
    runtime = {"train_seconds": round(time.time() - t_train, 1), "total_seconds": round(time.time() - t_start, 1)}
    try:
        import torch

        if torch.cuda.is_available():
            runtime["gpu"] = torch.cuda.get_device_name(0)
            runtime["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except ImportError:
        pass
    extra = {"run_name": config.run_name, "model_name": config.model_name, "backend": config.backend,
             "training_unit": "file",
             "dataset": {"version": meta.get("version"), "fingerprint": meta.get("fingerprint"),
                         "test_fold": config.test_fold, "val_fold": config.val_fold,
                         "files": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
                         "max_samples": config.max_samples},
             "input": window_info, "runtime": runtime, "file_level": res["info"]}
    return summarize(res["test_logits"], Y["test"], tax, config.output_dir, optimize=config.optimize_thresholds,
                     strategy=config.threshold_strategy, temperature_scaling=config.temperature_scaling,
                     val_logits=res["val_logits"], val_labels=Y["val"], test_exposed=exposed_test,
                     test_row_ids=test_df["row_id"].to_numpy(), val_row_ids=val_df["row_id"].to_numpy(),
                     save_logits=config.save_logits, extra=extra)


def _check_reproduced(metric: str, best, now: Dict[str, float], tol: float = 0.01) -> None:
    """The kept model must reproduce, on val, the score it was selected with.

    A mismatch means window predictions came back in a different order than the
    windows (e.g. a length-grouped eval sampler) or the best weights were not
    restored -- either way every file-level number would be wrong, so stop.
    """
    if best is None or metric not in now:
        LOGGER.warning("Consistency check skipped (best %s unknown)", metric)
        return
    if abs(float(now[metric]) - float(best)) > tol:
        raise RuntimeError(f"val {metric} after training ({now[metric]:.4f}) does not reproduce the selected "
                           f"epoch's value ({float(best):.4f}); predictions are misaligned with files or the "
                           "best weights were not restored -- refusing to write results")
    LOGGER.info("Consistency check passed: val %s after training %.4f == selected epoch %.4f",
                metric, now[metric], float(best))


def _val_metrics(window_logits: np.ndarray, ws_val: WindowSet, Y_val: np.ndarray) -> Dict[str, float]:
    file_logits = aggregate_max(window_logits, ws_val.file_idx, ws_val.n_files)
    probs = sigmoid(file_logits)
    at05 = prf_at(probs, Y_val, 0.5)
    return {"file_macro_ap": macro_average_precision(Y_val, probs),
            "file_f1_macro_05": at05["f1_macro"], "file_f1_micro_05": at05["f1_micro"],
            "window_f1_micro_05": prf_at(sigmoid(window_logits), ws_val.labels, 0.5)["f1_micro"]}


# --------------------------------------------------------------------------- #
# HuggingFace path (encoder / decoder)
# --------------------------------------------------------------------------- #
def _train_hf(config: TrainConfig, tax, tokenizer, windower, ws, train_idx, Y_val):
    import torch
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

    from .callbacks import make_progress_callback
    from .losses import make_asl_trainer_cls
    from .models import build_model

    model = build_model(config, tokenizer, tax)
    if config.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    train_ds = datamod.to_hf_dataset(ws["train"], train_idx)
    val_ds = datamod.to_hf_dataset(ws["val"])
    collator = datamod.PadCollator(windower.pad_id)
    eff = config.batch_size * config.gradient_accumulation_steps
    steps_per_epoch = max(math.ceil(len(train_ds) / eff), 1)
    total_steps = steps_per_epoch * config.num_epochs
    LOGGER.info("Effective batch %d | %d steps/epoch | %d total steps", eff, steps_per_epoch, total_steps)

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        preds = preds[0] if isinstance(preds, (tuple, list)) else preds
        return _val_metrics(np.asarray(preds, dtype=np.float32), ws["val"], Y_val)

    args = TrainingArguments(**_training_args_kwargs(dict(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=config.select_metric,
        greater_is_better=True,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=config.gradient_checkpointing,
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        disable_tqdm=True,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=config.dataloader_num_workers,
        seed=config.seed,
        data_seed=config.seed,
    ), warmup_ratio=config.warmup_ratio))
    callbacks = [EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience),
                 make_progress_callback(LOGGER, total_steps)]
    kwargs = dict(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator,
                  compute_metrics=compute_metrics, callbacks=callbacks)
    if config.loss_type == "asl":
        trainer = make_asl_trainer_cls()(asl_gamma_neg=config.asl_gamma_neg, asl_gamma_pos=config.asl_gamma_pos,
                                         asl_clip=config.asl_clip, **kwargs)
    else:
        trainer = Trainer(**kwargs)
    try:  # our callback already logs progress; drop the raw dict printer
        from transformers.trainer_callback import PrinterCallback

        trainer.remove_callback(PrinterCallback)
    except ImportError:
        pass

    resume = _find_checkpoint(config.output_dir) if config.resume_from_checkpoint else None
    if resume:
        LOGGER.info("Resuming from %s", resume)
    trainer.train(resume_from_checkpoint=resume)
    LOGGER.info("Best checkpoint: %s (%s=%.4f)", trainer.state.best_model_checkpoint, config.select_metric,
                trainer.state.best_metric if trainer.state.best_metric is not None else float("nan"))

    final_dir = Path(config.output_dir) / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    LOGGER.info("Saved final model to %s", final_dir)

    trainer.compute_metrics = None  # the val-file metric closure must not run on other splits

    def predict(w: WindowSet) -> np.ndarray:
        p = trainer.predict(datamod.to_hf_dataset(w)).predictions
        p = p[0] if isinstance(p, (tuple, list)) else p
        p = np.asarray(p, dtype=np.float32)
        if len(p) != len(w):
            raise RuntimeError(f"got {len(p)} predictions for {len(w)} windows")
        return p

    return predict, trainer.state.best_metric


def _training_args_kwargs(kw: Dict, warmup_ratio: float) -> Dict:
    """Adapt to the installed transformers (SCenv has 5.5.4, where warmup_ratio is deprecated
    in favour of a float warmup_steps).

    Length-grouped batching is deliberately NOT used: in transformers 5.5.4
    ``train_sampling_strategy="group_by_length"`` also puts a LengthGroupedSampler on the
    EVALUATION dataloader, so predictions come back in a shuffled order and the
    window -> file mapping (and every file-level metric) is silently scrambled.
    """
    import dataclasses

    import transformers
    from transformers import TrainingArguments

    fields = {f.name for f in dataclasses.fields(TrainingArguments)}
    if int(transformers.__version__.split(".")[0]) >= 5:
        kw["warmup_steps"] = float(warmup_ratio)            # < 1 -> ratio of total steps
    else:
        kw["warmup_ratio"] = warmup_ratio
    missing = sorted(set(kw) - fields)
    if missing:
        raise TypeError(f"transformers {transformers.__version__} TrainingArguments lacks {missing}")
    return kw


def _find_checkpoint(output_dir: str):
    ckpts = sorted(Path(output_dir).glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else -1)
    return str(ckpts[-1]) if ckpts else None


# --------------------------------------------------------------------------- #
# Native loop (TextCNN, OpenMythos)
# --------------------------------------------------------------------------- #
def _predict_native(model, w: WindowSet, pad_id: int, batch_size: int, device) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    order = np.argsort([len(x) for x in w.input_ids], kind="stable")  # length-sorted: less padding
    loader = DataLoader(datamod.WindowTorchDataset(w, order), batch_size=batch_size, shuffle=False,
                        collate_fn=datamod.PadCollator(pad_id))
    outs = []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            outs.append(logits.float().cpu().numpy())
    res = np.zeros((len(w), outs[0].shape[1] if outs else 0), dtype=np.float32)
    res[order] = np.concatenate(outs) if outs else res
    return res


def _native_loop(config: TrainConfig, model, pad_id: int, ws, train_idx, Y_val, loss_fn, optimizer,
                 use_schedule: bool):
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    g = torch.Generator()
    g.manual_seed(config.seed)
    loader = DataLoader(datamod.WindowTorchDataset(ws["train"], train_idx), batch_size=config.batch_size,
                        shuffle=True, collate_fn=datamod.PadCollator(pad_id), generator=g,
                        num_workers=config.dataloader_num_workers)
    gas = max(1, config.gradient_accumulation_steps)
    opt_steps_per_epoch = max(1, math.ceil(len(loader) / gas))
    total = opt_steps_per_epoch * config.num_epochs
    scheduler = None
    if use_schedule:
        warm = int(total * config.warmup_ratio)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    LOGGER.info("Native loop on %s | %d windows/epoch | batch %d x accum %d | %d optimizer steps", device,
                len(train_idx), config.batch_size, gas, total)

    best, best_state, bad_epochs, step = -1.0, None, 0, 0
    for epoch in range(config.num_epochs):
        model.train()
        running, n = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = loss_fn(logits.float(), batch["labels"].to(device)) / gas
            loss.backward()
            running += loss.item() * gas
            n += 1
            if (i + 1) % gas == 0 or (i + 1) == len(loader):
                if config.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    LOGGER.info("Step %d/%d (%.1f%%) | loss %.4f", step, total, 100.0 * step / total, running / n)
        vm = _val_metrics(_predict_native(model, ws["val"], pad_id, config.batch_size * 2, device), ws["val"], Y_val)
        LOGGER.info("EPOCH %d | train loss %.4f | val file macro-AP %.4f | file F1-macro@0.5 %.4f", epoch + 1,
                    running / max(n, 1), vm["file_macro_ap"], vm["file_f1_macro_05"])
        score = vm.get(config.select_metric, vm["file_macro_ap"])
        if score > best:
            best, bad_epochs = score, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= config.early_stopping_patience:
                LOGGER.info("Early stopping after epoch %d (no improvement for %d epochs)", epoch + 1, bad_epochs)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    LOGGER.info("Best val %s = %.4f", config.select_metric, best)

    def predict(w: WindowSet) -> np.ndarray:
        return _predict_native(model, w, pad_id, config.batch_size * 2, device)

    return predict, (best if best_state is not None else None)


def _save_safetensors(model, path: Path, meta: Dict) -> None:
    from safetensors.torch import save_file

    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if not v.is_complex()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path), metadata={k: json.dumps(v) for k, v in meta.items()})


def _train_textcnn(config: TrainConfig, tax, windower: RegexWindower, ws, train_idx, Y_val):
    import torch
    import torch.nn as nn

    from .textcnn import TextCNN

    model = TextCNN(len(windower.vocab), config.embed_dim, config.num_filters, config.filter_sizes,
                    tax.num_classes, config.cnn_dropout)
    # same optimisation as the scvd TextCNN baseline: Adam + BCE, constant LR
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    predict, best = _native_loop(config, model, windower.pad_id, ws, train_idx, Y_val, nn.BCEWithLogitsLoss(),
                                 optimizer, use_schedule=False)
    final = Path(config.output_dir) / "final_model"
    _save_safetensors(model, final / "model.safetensors", {"classes": list(tax.swc_ids)})
    (final / "vocab.json").write_text(json.dumps(windower.vocab), encoding="utf-8")
    return predict, best


def _train_openmythos(config: TrainConfig, tax, tokenizer, windower, ws, train_idx, Y_val):
    import torch

    from .losses import make_loss
    from .mythos import build_mythos

    model, info = build_mythos(config, tax.num_classes, windower.pad_id, len(tokenizer))
    if config.max_length > info["max_seq_len"]:
        raise ValueError(f"max_length {config.max_length} > OpenMythos max_seq_len {info['max_seq_len']}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = make_loss(config.loss_type, config.asl_gamma_neg, config.asl_gamma_pos, config.asl_clip)
    predict, best = _native_loop(config, model, windower.pad_id, ws, train_idx, Y_val, loss_fn, optimizer,
                                 use_schedule=True)
    final = Path(config.output_dir) / "final_model"
    _save_safetensors(model, final / "model.safetensors", {"classes": list(tax.swc_ids), "mythos": info})
    tokenizer.save_pretrained(str(final))
    return predict, best, {"openmythos": info}
