"""v2 invariants (file-level training, blending, CV summary). CPU only, seconds.
Run from the DAppSCAN_Training folder:  python tests/test_v2.py   (also pytest-compatible)."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scvd_dapp.filelevel import (AttentionPool, epoch_files, file_batches, group_windows,  # noqa: E402
                                 pos_weights, weighted_bce)
from scvd_dapp.windows import WindowSet  # noqa: E402


def _ws(file_idx, n_files):
    file_idx = np.asarray(file_idx)
    return WindowSet([np.arange(3) for _ in file_idx], np.zeros((len(file_idx), 2), np.float32), file_idx,
                     np.zeros((len(file_idx), 2), np.int64), np.zeros(len(file_idx), bool), n_files, {})


def test_group_windows_is_exact_and_ordered():
    g = group_windows(_ws([2, 0, 2, 1, 0, 2], 3))
    assert [list(x) for x in g] == [[1, 4], [3], [0, 2, 5]]
    try:
        group_windows(_ws([0, 0], 2))
    except ValueError:
        return
    raise AssertionError("a file without windows must be an error")


def test_epoch_files_keeps_every_labelled_file_and_thins_the_rest():
    Y = np.zeros((40, 2))
    Y[[0, 1], 0] = 1          # class 0: 2 files
    Y[[2], 1] = 1             # class 1: 1 file
    rng = np.random.RandomState(0)
    files = epoch_files(Y, neg_ratio=0.3, threshold=5, target=10, max_factor=3.0, rng=rng)
    uniq = set(files.tolist())
    assert {0, 1, 2} <= uniq
    neg = [f for f in uniq if Y[f].sum() == 0]
    assert len(neg) == round(37 * 0.3)
    # class 1: 1 file -> +min(10-1, 2*1)=2 copies; class 0: 2 files -> +min(8, 4)=4 copies
    assert (files == 2).sum() == 1 + 2 + 0 and sum((files == i).sum() for i in (0, 1)) == 2 + 4
    # a fresh draw of negatives every epoch
    files2 = epoch_files(Y, 0.3, 5, 10, 3.0, rng)
    assert set(f for f in files2 if Y[f].sum() == 0) != set(neg)


def test_pos_weights_and_weighted_bce_match_torch():
    import torch

    Y = np.array([[1, 0], [0, 0], [0, 1], [0, 0]], dtype=np.float32)
    pw = pos_weights(Y, np.arange(4), cap=10.0)
    assert np.allclose(pw, [3.0, 3.0])
    assert np.allclose(pos_weights(Y, np.arange(4), cap=2.0), [2.0, 2.0])
    x = np.array([[2.0, -1.0], [-0.5, 0.3], [0.1, 1.5], [-3.0, -2.0]], dtype=np.float32)
    ref = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw))(torch.tensor(x), torch.tensor(Y)).item()
    assert abs(weighted_bce(x, Y, pw) - ref) < 1e-6


def test_file_batches_keep_files_whole_and_cap_long_files():
    groups = [np.arange(0, 3), np.arange(3, 23), np.arange(23, 24), np.arange(24, 26)]
    batches = list(file_batches(np.array([0, 1, 2, 3]), groups, max_windows=8, window_batch=6,
                                rng=np.random.RandomState(0)))
    seen = [f for b in batches for f, _ in b]
    assert seen == [0, 1, 2, 3]
    for b in batches:
        for f, w in b:
            assert set(w) <= set(groups[f]) and len(w) == min(len(groups[f]), 8)
        assert len(b) == 1 or sum(len(w) for _, w in b) <= 6


def test_attention_pool_is_order_invariant_and_normalised():
    import torch

    torch.manual_seed(0)
    head = AttentionPool(16, 3, att_dim=8, dropout=0.0).eval()
    H = torch.randn(5, 16)
    out = head(H)
    assert out.shape == (3,)
    perm = torch.randperm(5)
    assert torch.allclose(out, head(H[perm]), atol=1e-5)
    att = head.attention(H)
    assert torch.allclose(att.sum(0), torch.ones(3), atol=1e-5)
    # a long file made of copies of the same window scores like the single window (no length bonus)
    assert torch.allclose(head(H[:1]), head(H[:1].repeat(7, 1)), atol=1e-5)


def _fake_run(d: Path, logits_val, logits_test, Yv, Yt):
    d.mkdir(parents=True, exist_ok=True)
    rv, rt = np.arange(len(Yv)), np.arange(len(Yt)) + 1000
    np.savez_compressed(d / "logits_val.npz", logits=logits_val.astype(np.float32), labels=Yv.astype(np.uint8),
                        row_id=rv, classes=np.asarray(["SWC-135", "SWC-101"]))
    np.savez_compressed(d / "logits_test.npz", logits=logits_test.astype(np.float32), labels=Yt.astype(np.uint8),
                        row_id=rt, exposed=np.zeros(0, np.uint8), classes=np.asarray(["SWC-135", "SWC-101"]))
    (d / "test_results.json").write_text(json.dumps({"dataset": {"fingerprint": "x"}}), encoding="utf-8")


def test_blend_prefers_the_informative_run():
    from scvd_dapp import blend as blendmod
    from scvd_dapp.taxonomy import Taxonomy

    rng = np.random.RandomState(0)
    n = 600
    Yv = (rng.rand(n, 2) < 0.1).astype(np.float32)
    Yt = (rng.rand(n, 2) < 0.1).astype(np.float32)
    good_v, good_t = Yv * 2.0 + rng.randn(n, 2), Yt * 2.0 + rng.randn(n, 2)
    noise_v, noise_t = rng.randn(n, 2), rng.randn(n, 2)
    tax = Taxonomy(("SWC-135", "SWC-101"))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _fake_run(tmp / "good", good_v, good_t, Yv, Yt)
        _fake_run(tmp / "noise", noise_v, noise_t, Yv, Yt)
        import scvd_dapp.taxonomy as taxmod

        saved = taxmod.load_taxonomy
        taxmod.load_taxonomy = lambda *_a, **_k: tax
        try:
            res = blendmod.run_blend(str(tmp / "good"), str(tmp / "noise"), str(tmp / "blend"))
        finally:
            taxmod.load_taxonomy = saved
        assert res["blend"]["w_a"] >= 0.7, res["blend"]
        assert res["macro_ap"] > 0.5
        # mismatched files must be refused
        _fake_run(tmp / "other", noise_v[:-1], noise_t, Yv[:-1], Yt)
        try:
            blendmod.run_blend(str(tmp / "good"), str(tmp / "other"), str(tmp / "blend2"))
        except ValueError:
            pass
        else:
            raise AssertionError("runs on different files must not be blended")


def test_cv_summary_groups_folds():
    from scvd_dapp.collect import collect_cv

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, vals in (("modernbert", [0.05, 0.07]), ("tfidf", [0.06, 0.05])):
            for k, v in enumerate(vals):
                d = tmp / f"{name}_f{k}"
                d.mkdir()
                (d / "test_results.json").write_text(json.dumps({"f1": v, "macro_ap": v / 2, "dataset": {}}))
        text = collect_cv(str(tmp))
        assert "| modernbert | 0,1 | 0.0600" in text and "| tfidf | 0,1 | 0.0550" in text
        assert (tmp / f"RESULTS_{tmp.name}.md").exists()


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
