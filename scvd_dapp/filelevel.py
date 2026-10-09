"""v2: file-level training -- one label vector per FILE; windows pooled by per-class attention.

Why (from the v1 results; see RUNBOOK "v2"):

* v1 trained on WINDOW labels and scored a file by its highest window. Every transformer
  peaked in its first epochs and then over-fit, and a plain TF-IDF + logistic regression on
  whole files beat all of them (fold 0: F1 0.058 vs 0.031 for the best transformer).
* max-over-windows ties a file's score to its number of windows, i.e. to its length.

What v2 changes (all configurable in configs/*_v2.yaml):

* The model is trained on exactly what it is evaluated on: a file and its labels. A file's
  windows (at most ``max_windows_train`` per step while training -- a fresh random subset each
  epoch -- and all of them at evaluation) are encoded, and a per-class gated attention
  (Ilse et al., 2018) pools them into one vector per class before that class's classifier.
* Regularisation: the bottom ``freeze_frac`` of an encoder's layers are frozen (decoders: LoRA
  on the top layers only); a low backbone learning rate and a separate one for the new head;
  dropout on the pooled vector; early stopping on the validation LOSS (smooth), where v1
  selected on a noisy validation average precision.
* Balancing, per epoch: every labelled training file, a fresh ``neg_ratio`` draw of the
  unlabelled ones, and rare classes' files duplicated towards ``file_oversample_target`` (at
  most ``oversample_max_factor`` copies); BCE with a per-class ``pos_weight`` (negatives /
  positives in the epoch sample, capped at ``pos_weight_cap``).

Validation and test are never resampled, and F1 is computed exactly as in v1
(``evaluate.summarize``): per-class thresholds tuned on validation, macro over classes.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .metrics import macro_average_precision, prf_at, sigmoid
from .windows import WindowSet

LOGGER = logging.getLogger("scvd_dapp")


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested in tests/test_v2.py)
# --------------------------------------------------------------------------- #
def group_windows(ws: WindowSet) -> List[np.ndarray]:
    """Window indices of every file, in file order. Every file must have >= 1 window."""
    order = np.argsort(ws.file_idx, kind="stable")
    bounds = np.searchsorted(ws.file_idx[order], np.arange(ws.n_files + 1))
    groups = [order[bounds[i]:bounds[i + 1]] for i in range(ws.n_files)]
    empty = [i for i, g in enumerate(groups) if len(g) == 0]
    if empty:
        raise ValueError(f"{len(empty)} files have no windows (first: {empty[:5]})")
    return groups


def epoch_files(Y: np.ndarray, neg_ratio: float, threshold: int, target: int, max_factor: float,
                rng: np.random.RandomState) -> np.ndarray:
    """Training files for one epoch, shuffled: every labelled file, a fresh ``neg_ratio`` draw of
    the unlabelled ones, and duplicates of rare classes' files (towards ``target`` files, never more
    than ``max_factor`` x a class's count)."""
    pos = np.flatnonzero(Y.sum(1) > 0)
    neg = np.flatnonzero(Y.sum(1) == 0)
    n_neg = int(round(len(neg) * neg_ratio)) if neg_ratio < 1.0 else len(neg)
    parts = [pos, rng.choice(neg, size=n_neg, replace=False) if n_neg < len(neg) else neg]
    counts = Y[pos].sum(0)
    for k in np.argsort(counts, kind="stable"):
        c = int(counts[k])
        if c == 0 or c >= threshold:
            continue
        need = int(min(target - c, (max_factor - 1.0) * c))
        if need > 0:
            parts.append(rng.choice(pos[Y[pos, k] > 0], size=need, replace=True))
    out = np.concatenate(parts).astype(np.int64)
    rng.shuffle(out)
    return out


def pos_weights(Y: np.ndarray, files: np.ndarray, cap: float) -> np.ndarray:
    """Per-class BCE pos_weight = negatives / positives among ``files``, clipped to [1, cap]."""
    sub = Y[files]
    pos = sub.sum(0)
    neg = len(files) - pos
    return np.clip(neg / np.maximum(pos, 1.0), 1.0, cap).astype(np.float32)


def weighted_bce(logits: np.ndarray, Y: np.ndarray, pw: np.ndarray) -> float:
    """Mean BCE-with-logits with per-class pos_weight (same as the training loss)."""
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(Y, dtype=np.float64)
    log_p = -np.logaddexp(0.0, -x)          # log sigmoid(x)
    log_1mp = -np.logaddexp(0.0, x)         # log(1 - sigmoid(x))
    return float((-(pw[None, :] * y * log_p + (1.0 - y) * log_1mp)).mean())


def file_batches(files: np.ndarray, groups: List[np.ndarray], max_windows: int, window_batch: int,
                 rng: np.random.RandomState) -> Iterator[List[Tuple[int, np.ndarray]]]:
    """Group whole files into forward passes of <= ``window_batch`` windows (a file with more
    windows than that goes alone). Files longer than ``max_windows`` get a random subset."""
    cur: List[Tuple[int, np.ndarray]] = []
    n = 0
    for f in files:
        w = groups[int(f)]
        if len(w) > max_windows:
            w = np.sort(rng.choice(w, size=max_windows, replace=False))
        if cur and n + len(w) > window_batch:
            yield cur
            cur, n = [], 0
        cur.append((int(f), w))
        n += len(w)
    if cur:
        yield cur


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class AttentionPool(nn.Module):
    """Per-class gated attention MIL pooling (Ilse et al., 2018) and one linear classifier per class.

    H (n_windows, d) -> logits (K,). Each class attends to the windows that matter for it, so a
    long file is not favoured just for having more windows (v1's max was)."""

    def __init__(self, d: int, num_classes: int, att_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.V = nn.Linear(d, att_dim)
        self.U = nn.Linear(d, att_dim)
        self.w = nn.Linear(att_dim, num_classes)
        self.drop = nn.Dropout(dropout)
        self.cls_w = nn.Parameter(torch.randn(num_classes, d) * 0.02)
        self.cls_b = nn.Parameter(torch.zeros(num_classes))

    def attention(self, H: torch.Tensor) -> torch.Tensor:
        H = self.norm(H)
        return torch.softmax(self.w(torch.tanh(self.V(H)) * torch.sigmoid(self.U(H))), dim=0)  # (n, K)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        Hn = self.norm(H)
        att = torch.softmax(self.w(torch.tanh(self.V(Hn)) * torch.sigmoid(self.U(Hn))), dim=0)
        Z = att.transpose(0, 1) @ self.drop(Hn)                      # (K, d)
        return (Z * self.cls_w).sum(-1) + self.cls_b                 # (K,)


class FileModel(nn.Module):
    """Backbone window encoder + attention pooling over a file's windows."""

    def __init__(self, backbone: nn.Module, hidden: int, num_classes: int, is_decoder: bool,
                 att_dim: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        self.is_decoder = is_decoder
        self.hidden = hidden
        self.head = AttentionPool(hidden, num_classes, att_dim, dropout)

    def embed(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(b, T) windows -> (b, d) float32. Encoders: masked mean; decoders: last real token."""
        out = self.backbone(input_ids=ids, attention_mask=mask)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        if self.is_decoder:
            last = mask.sum(1).long() - 1                            # right padding
            e = h[torch.arange(h.size(0), device=h.device), last]
        else:
            m = mask.unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return e.float()

    def pool(self, H: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=H.device.type, enabled=False):
            return self.head(H.float())


def _layer_list(backbone: nn.Module, n_layers: int) -> nn.ModuleList:
    for _, mod in backbone.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == n_layers:
            return mod
    raise RuntimeError(f"could not find the list of {n_layers} transformer layers in {type(backbone).__name__}")


def freeze_bottom(backbone: nn.Module, k: int) -> None:
    """Freeze the embeddings and the first ``k`` transformer layers."""
    if k <= 0:
        return
    layers = _layer_list(backbone, backbone.config.num_hidden_layers)
    for p in backbone.get_input_embeddings().parameters():
        p.requires_grad = False
    emb = getattr(backbone, "embeddings", None)
    if isinstance(emb, nn.Module):
        for p in emb.parameters():
            p.requires_grad = False
    for layer in layers[:k]:
        for p in layer.parameters():
            p.requires_grad = False


def _merge_adapter(base_name: str, adapter_dir: Path, tokenizer, kwargs: Dict) -> nn.Module:
    """v3: merge a PEFT LoRA adapter trained on ``AutoModelForSequenceClassification(base)`` --
    the Slither decoders -- into the base weights and return the bare transformer (the
    Slither head is dropped). The base is the one the adapter was trained on."""
    from peft import PeftConfig, PeftModel
    from safetensors import safe_open
    from transformers import AutoModelForSequenceClassification

    from .env import load_with_offline_fallback

    trained_on = PeftConfig.from_pretrained(str(adapter_dir)).base_model_name_or_path
    if trained_on and trained_on != base_name:
        LOGGER.warning("adapter %s was trained on %s, config model_name is %s -- merging into %s",
                       adapter_dir, trained_on, base_name, trained_on)
        base_name = trained_on
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt") as fh:
        keys = list(fh.keys())
        score = [k for k in keys if k.endswith("score.weight")]
        emb = [k for k in keys if k.endswith("embed_tokens.weight")]
        num_labels = int(fh.get_slice(score[0]).get_shape()[0]) if score else 2
        vocab = int(fh.get_slice(emb[0]).get_shape()[0]) if emb else None
    clf = load_with_offline_fallback(AutoModelForSequenceClassification.from_pretrained, base_name,
                                     num_labels=num_labels, **kwargs)
    if vocab is None and tokenizer is not None and len(tokenizer) > clf.get_input_embeddings().weight.shape[0]:
        vocab = len(tokenizer)               # adapter saved without its (resized) embeddings
    if vocab is not None and clf.get_input_embeddings().weight.shape[0] != vocab:
        clf.resize_token_embeddings(vocab)
    if tokenizer is not None and tokenizer.pad_token_id is not None:
        clf.config.pad_token_id = tokenizer.pad_token_id
    merged = PeftModel.from_pretrained(clf, str(adapter_dir)).merge_and_unload()
    backbone = getattr(merged, merged.base_model_prefix)
    LOGGER.info("Backbone = %s with the LoRA adapter %s merged in (%d-way head dropped, vocab %s)",
                base_name, adapter_dir, num_labels, vocab)
    return backbone


def load_backbone(config, tokenizer, is_dec: bool, kwargs: Dict) -> nn.Module:
    """The public base model, or (v3, ``backbone_init``) a backbone that starts from a fine-tuned
    checkpoint: a full HF checkpoint (encoders) or a PEFT adapter (decoders)."""
    from transformers import AutoModel

    from .env import load_with_offline_fallback

    if not config.backbone_init:
        LOGGER.info("Loading backbone %s (%s)", config.model_name, "decoder" if is_dec else "encoder")
        return load_with_offline_fallback(AutoModel.from_pretrained, config.model_name, **kwargs)
    path = Path(config.backbone_init)
    if not path.is_dir():
        raise FileNotFoundError(f"backbone_init {path} is not a directory")
    if (path / "adapter_config.json").exists():
        return _merge_adapter(config.model_name, path, tokenizer, kwargs)
    LOGGER.info("Loading backbone from the fine-tuned checkpoint %s (%s)", path, "decoder" if is_dec else "encoder")
    return AutoModel.from_pretrained(str(path), **kwargs)


def build_file_model(config, tokenizer, num_classes: int) -> Tuple[FileModel, Dict]:
    from .config import BACKEND_HF_DECODER

    is_dec = config.backend == BACKEND_HF_DECODER
    kwargs = {"trust_remote_code": True}
    if is_dec and torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16
    backbone = load_backbone(config, tokenizer, is_dec, kwargs)
    if len(tokenizer) > backbone.get_input_embeddings().weight.shape[0]:
        backbone.resize_token_embeddings(len(tokenizer))
    n_layers = int(backbone.config.num_hidden_layers)
    hidden = int(backbone.config.hidden_size)
    k_frozen = int(math.floor(n_layers * config.freeze_frac))
    if hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False
    if config.gradient_checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if is_dec and config.use_lora:
        from peft import LoraConfig, get_peft_model

        targets = list(config.lora_target_modules)
        if k_frozen:
            # LoRA on the top layers only. A regex over full module names (fullmatch), because
            # peft's layers_to_transform expects a "model.layers.N" prefix that a bare AutoModel lacks.
            top = "|".join(str(i) for i in range(k_frozen, n_layers))
            targets = rf"(?:.*\.)?layers\.(?:{top})\.(?:.*\.)?(?:{'|'.join(targets)})"
        lora = LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                          target_modules=targets, bias="none")
        backbone = get_peft_model(backbone, lora)
        n_lora = sum(1 for n, _ in backbone.named_modules() if n.endswith("lora_A"))
        LOGGER.info("LoRA on layers %d-%d (%d adapted projections)", k_frozen, n_layers - 1, n_lora)
        for p in backbone.parameters():          # LoRA weights train in fp32 under bf16 autocast
            if p.requires_grad:
                p.data = p.data.float()
    else:
        freeze_bottom(backbone, k_frozen)
    model = FileModel(backbone, hidden, num_classes, is_dec, config.att_dim, config.head_dropout)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    info = {"layers": n_layers, "frozen_or_lora_free_layers": k_frozen, "hidden": hidden,
            "trainable_params": int(trainable), "total_params": int(total),
            "lora": bool(is_dec and config.use_lora), "backbone_init": config.backbone_init}
    LOGGER.info("File-level model: %d layers (%d bottom %s), %d / %d trainable (%.2f%%)", n_layers, k_frozen,
                "without LoRA" if info["lora"] else "frozen", trainable, total, 100.0 * trainable / max(total, 1))
    return model, info


# --------------------------------------------------------------------------- #
# Forward helpers
# --------------------------------------------------------------------------- #
def _pad(windows: List[np.ndarray], pad_id: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    n = max(len(w) for w in windows)
    n = ((n + 7) // 8) * 8
    ids = torch.full((len(windows), n), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(windows), n), dtype=torch.long)
    for i, w in enumerate(windows):
        ids[i, : len(w)] = torch.as_tensor(np.asarray(w), dtype=torch.long)
        mask[i, : len(w)] = 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def predict_files(model: FileModel, ws: WindowSet, groups: List[np.ndarray], pad_id: int, batch: int,
                  device, use_amp: bool) -> np.ndarray:
    """File logits (n_files, K) from ALL of each file's windows."""
    model.eval()
    order = np.argsort([len(x) for x in ws.input_ids], kind="stable")     # less padding
    E = torch.empty((len(ws), model.hidden), dtype=torch.float32)
    for i in range(0, len(order), batch):
        idx = order[i:i + batch]
        ids, mask = _pad([ws.input_ids[j] for j in idx], pad_id, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            E[torch.as_tensor(idx)] = model.embed(ids, mask).cpu()
    out = np.zeros((ws.n_files, model.head.cls_b.numel()), dtype=np.float32)
    for f, g in enumerate(groups):
        out[f] = model.pool(E[torch.as_tensor(g)].to(device)).float().cpu().numpy()
    return out


def _trainable_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if p.requires_grad}


def _restore(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in state.items():
            params[n].copy_(v.to(device=params[n].device, dtype=params[n].dtype))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_file_level(config, tax, tokenizer, windower, ws: Dict[str, WindowSet], Y: Dict[str, np.ndarray]) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    torch.manual_seed(config.seed)
    rng = np.random.RandomState(config.seed)
    K = tax.num_classes
    pad_id = windower.pad_id

    model, info = build_file_model(config, tokenizer, K)
    model.to(device)
    ev = np.asarray(tax.eval_indices())   # v3: early stopping and logs use the scored classes only
    groups = {s: group_windows(ws[s]) for s in ("train", "val", "test")}

    head_ids = {id(p) for p in model.head.parameters()}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": backbone_params, "lr": config.learning_rate},
                             {"params": list(model.head.parameters()), "lr": config.head_learning_rate}],
                            weight_decay=config.weight_decay)

    probe = epoch_files(Y["train"], config.neg_ratio, config.file_oversample_threshold,
                        config.file_oversample_target, config.oversample_max_factor,
                        np.random.RandomState(config.seed + 1))
    pw_np = pos_weights(Y["train"], probe, config.pos_weight_cap)
    pw = torch.as_tensor(pw_np, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="sum")
    steps_per_epoch = max(1, math.ceil(len(probe) / config.files_per_step))
    total = steps_per_epoch * config.num_epochs
    warm = int(total * config.warmup_ratio)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    n_pos = int((Y["train"].sum(1) > 0).sum())
    LOGGER.info("v2 file-level training on %s | %d train files (%d labelled) | ~%d files/epoch | %d files/step | "
                "%d steps/epoch | max %d windows/file | pos_weight %s", device, len(Y["train"]), n_pos, len(probe),
                config.files_per_step, steps_per_epoch, config.max_windows_train,
                np.array2string(pw_np, precision=1, max_line_width=200))

    best, best_state, best_epoch, bad, step = math.inf, None, 0, 0, 0
    history: List[Dict] = []
    t0 = time.time()
    for epoch in range(config.num_epochs):
        model.train()
        files = epoch_files(Y["train"], config.neg_ratio, config.file_oversample_threshold,
                            config.file_oversample_target, config.oversample_max_factor, rng)
        run_sum, run_n, since = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for batch in file_batches(files, groups["train"], config.max_windows_train, config.window_batch, rng):
            wins = [ws["train"].input_ids[j] for _, w in batch for j in w]
            ids, mask = _pad(wins, pad_id, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                E = model.embed(ids, mask)
            offs = np.cumsum([0] + [len(w) for _, w in batch])
            logits = torch.stack([model.pool(E[offs[i]:offs[i + 1]]) for i in range(len(batch))])
            yb = torch.as_tensor(Y["train"][[f for f, _ in batch]], dtype=torch.float32, device=device)
            loss_sum = loss_fn(logits, yb)
            (loss_sum / (K * config.files_per_step)).backward()
            run_sum += float(loss_sum.item())
            run_n += len(batch) * K
            since += len(batch)
            if since >= config.files_per_step:
                if config.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                   config.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                since = 0
                if step % 25 == 0:
                    LOGGER.info("Step %d/%d (%.1f%%) | train loss %.4f", step, total, 100.0 * step / total,
                                run_sum / max(run_n, 1))
        if since > 0:
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

        val_logits = predict_files(model, ws["val"], groups["val"], pad_id, config.eval_window_batch, device, use_amp)
        vloss = weighted_bce(val_logits[:, ev], Y["val"][:, ev], pw_np[ev])
        vprob = sigmoid(val_logits[:, ev])
        rec = {"epoch": epoch + 1, "train_loss": round(run_sum / max(run_n, 1), 5), "val_loss": round(vloss, 5),
               "val_macro_ap": round(macro_average_precision(Y["val"][:, ev], vprob), 5),
               "val_f1_macro_05": round(prf_at(vprob, Y["val"][:, ev], 0.5)["f1_macro"], 5),
               "minutes": round((time.time() - t0) / 60.0, 1)}
        history.append(rec)
        LOGGER.info("EPOCH %d | train loss %.4f | val loss %.4f | val file macro-AP %.4f | val F1-macro@0.5 %.4f",
                    rec["epoch"], rec["train_loss"], vloss, rec["val_macro_ap"], rec["val_f1_macro_05"])
        if vloss < best - 1e-4:
            best, best_epoch, bad = vloss, epoch + 1, 0
            best_state = _trainable_state(model)
        else:
            bad += 1
            if bad >= config.early_stopping_patience:
                LOGGER.info("Early stopping after epoch %d (val loss did not improve for %d epochs)", epoch + 1, bad)
                break

    if best_state is None:
        raise RuntimeError("no epoch produced a finite validation loss")
    _restore(model, best_state)
    val_logits = predict_files(model, ws["val"], groups["val"], pad_id, config.eval_window_batch, device, use_amp)
    now = weighted_bce(val_logits[:, ev], Y["val"][:, ev], pw_np[ev])
    if abs(now - best) > 1e-3:
        raise RuntimeError(f"val loss after restoring the best weights ({now:.5f}) does not reproduce the selected "
                           f"epoch's ({best:.5f}); refusing to write results")
    LOGGER.info("Consistency check passed: val loss after training %.5f == best epoch (%d) %.5f", now, best_epoch, best)
    test_logits = predict_files(model, ws["test"], groups["test"], pad_id, config.eval_window_batch, device, use_amp)

    final = Path(config.output_dir) / "final_model"
    final.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    save_file({n: v.contiguous() for n, v in best_state.items()}, str(final / "trainable.safetensors"),
              metadata={"base_model": config.model_name, "classes": json.dumps(list(tax.swc_ids))})
    tokenizer.save_pretrained(str(final))
    LOGGER.info("Saved the trained parameters (%d tensors) to %s", len(best_state), final)
    return {"val_logits": val_logits, "test_logits": test_logits,
            "info": {**info, "best_epoch": best_epoch, "best_val_loss": round(best, 5), "history": history,
                     "pos_weight": [round(float(x), 3) for x in pw_np], "steps": step}}
