"""v3 invariants: scored-class subset, safe-file balancing, backbone initialisation.
Run from the package folder (DAppSCAN_Training_v3 on the cluster):  python tests/test_v3.py
(also pytest-compatible).

The two backbone tests need the tiny test models in the local HuggingFace cache
(hf-internal-testing/tiny-random-ModernBertForSequenceClassification,
trl-internal-testing/tiny-Qwen3ForCausalLM); they are skipped when those are absent."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scvd_dapp.taxonomy import Taxonomy  # noqa: E402

TINY_ENCODER = "hf-internal-testing/tiny-random-ModernBertForSequenceClassification"
TINY_DECODER = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def test_taxonomy_reads_eval_classes():
    meta = {"classes": [{"index": i, "swc": s} for i, s in enumerate(["SWC-135", "SWC-101", "SWC-109"])],
            "eval_classes": [1, 0]}
    tax = Taxonomy.from_meta(meta)
    assert tax.num_classes == 3 and tax.eval_indices() == [0, 1]
    assert tax.subset([0, 1]).swc_ids == ("SWC-135", "SWC-101")
    meta.pop("eval_classes")
    assert Taxonomy.from_meta(meta).eval_indices() == [0, 1, 2]


def test_summarize_scores_only_eval_classes():
    from scvd_dapp.evaluate import summarize

    rng = np.random.RandomState(0)
    n = 300
    Y = (rng.rand(n, 3) < 0.2).astype(np.float32)
    logits = Y * 3.0 + rng.randn(n, 3)
    logits[:, 2] = rng.randn(n)              # class 2: pure noise, NOT scored
    tax = Taxonomy(("SWC-135", "SWC-101", "SWC-109"), (0, 1))
    with tempfile.TemporaryDirectory() as tmp:
        r = summarize(logits, Y, tax, tmp, val_logits=logits, val_labels=Y)
        assert r["n_classes"] == 2 and r["classes"] == ["SWC-135", "SWC-101"]
        assert r["trained_classes"] == ["SWC-135", "SWC-101", "SWC-109"]
        assert abs(r["f1"] - np.mean([c["f1"] for c in r["per_class"]])) < 1e-12
        z = np.load(Path(tmp) / "logits_test.npz")
        assert z["logits"].shape == (n, 3) and z["labels"].shape == (n, 3)   # blends need every class
        full = summarize(logits, Y, Taxonomy(tax.swc_ids), tmp + "/all", val_logits=logits, val_labels=Y)
        assert full["n_classes"] == 3 and full["f1"] < r["f1"]               # the noise class drags it down


def _fake_table(path: Path):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.RandomState(1)
    rows = []
    for i in range(400):
        fold = i % 5
        labs = [int(rng.randint(3))] if rng.rand() < 0.1 else []
        rows.append({"row_id": i, "content_id": f"c{i}", "fold": fold, "labels": labs,
                     "spans_json": "[]", "source_code": "contract A {}", "n_chars": 13, "n_lines": 1})
    df = pd.DataFrame(rows)
    meta = {"version": "fake_all", "fingerprint": "f", "params": {"n_folds": 5},
            "classes": [{"index": i, "swc": s, "title": "t", "n_rows": 1, "per_fold": [0] * 5}
                        for i, s in enumerate(["SWC-135", "SWC-101", "SWC-109"])],
            "eval_classes": [0, 1], "eval_min_files": 20, "counts": {"rows": len(df)}}
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table.replace_schema_metadata({b"dappscan_meta": json.dumps(meta).encode()}), path)
    path.with_suffix("").with_name(path.stem + ".meta.json").write_text(json.dumps(meta))
    pd.DataFrame({"row_a": [0, 1, 5], "row_b": [399, 2, 10], "jaccard": [0.9, 0.9, 0.9]}).to_csv(
        path.with_name(path.stem + ".neardup.csv"), index=False)
    return df


def test_balance_keeps_every_labelled_file_and_thins_the_rest_per_fold():
    import pandas as pd

    from scvd_dapp.balance import balance_table

    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "fake_all.parquet", Path(tmp) / "fake.parquet"
        df = _fake_table(src)
        meta = balance_table(str(src), str(out), safe_ratio=1.0, seed=0)
        bal = pd.read_parquet(out)
        lab = bal["labels"].apply(len) > 0
        assert set(bal[lab]["row_id"]) == set(df[df["labels"].apply(len) > 0]["row_id"])   # all labelled kept
        for f in range(5):
            in_f = bal["fold"] == f
            assert int((in_f & ~lab).sum()) == int((in_f & lab).sum())                    # 1 safe per labelled
        assert set(bal["row_id"]) <= set(df["row_id"])                                      # ids preserved
        assert meta["eval_classes"] == [0, 1] and meta["version"] == "fake"
        side = pd.read_csv(out.with_name("fake.neardup.csv"))
        assert set(side["row_a"]) | set(side["row_b"]) <= set(bal["row_id"])                # sidecar filtered
        assert Path(str(out.with_suffix("")) + ".DATA_CARD.md").exists()


def _have(model_id: str) -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_id, local_files_only=True)
        return True
    except Exception:
        return False


def _cfg(**kw):
    from types import SimpleNamespace

    return SimpleNamespace(**{"backbone_init": None, "model_name": None, **kw})


def test_encoder_starts_from_a_finetuned_checkpoint():
    if not _have(TINY_ENCODER):
        print("  (skipped: tiny encoder not cached)")
        return
    import torch
    from transformers import AutoModelForSequenceClassification

    from scvd_dapp.filelevel import load_backbone

    torch.manual_seed(0)
    clf = AutoModelForSequenceClassification.from_pretrained(TINY_ENCODER, num_labels=39,
                                                             ignore_mismatched_sizes=True)
    with torch.no_grad():                    # make it differ from the public weights
        for p in clf.parameters():
            p.add_(0.01 * torch.randn_like(p))
    with tempfile.TemporaryDirectory() as tmp:
        clf.save_pretrained(tmp)
        bb = load_backbone(_cfg(backbone_init=tmp, model_name=TINY_ENCODER), None, False, {})
        base = getattr(clf, clf.base_model_prefix)
        sd = base.state_dict()
        for k, v in bb.state_dict().items():
            assert torch.equal(v, sd[k]), k


def test_decoder_adapter_is_merged_exactly():
    if not _have(TINY_DECODER):
        print("  (skipped: tiny decoder not cached)")
        return
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from scvd_dapp.filelevel import load_backbone

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(TINY_DECODER)
    tok.add_special_tokens({"pad_token": "[PAD]"})      # like the Slither CodeLlama tokenizer
    clf = AutoModelForSequenceClassification.from_pretrained(TINY_DECODER, num_labels=39, dtype=torch.float32)
    clf.resize_token_embeddings(len(tok))
    clf.config.pad_token_id = tok.pad_token_id
    peft = get_peft_model(clf, LoraConfig(task_type=TaskType.SEQ_CLS, r=4, lora_alpha=8,
                                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                          modules_to_save=["score"]))
    with torch.no_grad():                    # non-zero LoRA B, so the merge changes the weights
        for n, p in peft.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.1)
    peft.eval()
    ids = torch.randint(5, 100, (2, 12))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        ref = peft.base_model.model.model(input_ids=ids, attention_mask=mask).last_hidden_state
    with tempfile.TemporaryDirectory() as tmp:
        peft.save_pretrained(tmp, save_embedding_layers=True)   # the Slither adapters carry their embeddings
        tok.save_pretrained(tmp)
        bb = load_backbone(_cfg(backbone_init=tmp, model_name=TINY_DECODER), tok, True,
                           {"dtype": torch.float32}).eval()   # fp32 so the check is exact (GPU runs use bf16)
        with torch.no_grad():
            got = bb(input_ids=ids, attention_mask=mask).last_hidden_state
    assert bb.get_input_embeddings().weight.shape[0] == len(tok)
    assert torch.allclose(got, ref, atol=1e-4), float((got - ref).abs().max())


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
