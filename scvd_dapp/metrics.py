"""Multi-label metrics, threshold tuning and calibration -- at FILE level.

Everything here takes logits of shape (n, K) and a 0/1 label matrix of the same
shape. K comes from the dataset taxonomy (never hardcoded).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .taxonomy import Taxonomy


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def prf_at(probs: np.ndarray, labels: np.ndarray, thresholds) -> Dict[str, float]:
    """Micro/macro precision-recall-F1 of ``probs >= thresholds``."""
    from sklearn.metrics import precision_recall_fscore_support

    preds = (probs >= np.asarray(thresholds)).astype(int)
    labels = labels.astype(int)
    p_mi, r_mi, f_mi, _ = precision_recall_fscore_support(labels, preds, average="micro", zero_division=0)
    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    exact = float((preds == labels).all(1).mean()) if len(labels) else 0.0
    return {"f1_micro": float(f_mi), "f1_macro": float(f_ma), "precision_micro": float(p_mi),
            "recall_micro": float(r_mi), "precision_macro": float(p_ma), "recall_macro": float(r_ma),
            "exact_match": exact}


def macro_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mean per-class average precision (PR-AUC) over classes with >= 1 positive. Threshold-free."""
    from sklearn.metrics import average_precision_score

    aps = [average_precision_score(labels[:, k], scores[:, k])
           for k in range(labels.shape[1]) if labels[:, k].sum() > 0]
    return float(np.mean(aps)) if aps else 0.0


def per_class_average_precision(labels: np.ndarray, scores: np.ndarray) -> List[float]:
    from sklearn.metrics import average_precision_score

    return [float(average_precision_score(labels[:, k], scores[:, k])) if labels[:, k].sum() > 0 else 0.0
            for k in range(labels.shape[1])]


def per_class_report(probs: np.ndarray, labels: np.ndarray, thresholds, tax: Taxonomy) -> List[Dict]:
    thr = np.full(labels.shape[1], thresholds) if np.isscalar(thresholds) else np.asarray(thresholds)
    preds = (probs >= thr).astype(int)
    aps = per_class_average_precision(labels, probs)
    rows = []
    for k in range(labels.shape[1]):
        y, p = labels[:, k].astype(int), preds[:, k]
        tp = int(((y == 1) & (p == 1)).sum())
        fp = int(((y == 0) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"label_id": k, "swc": tax.swc_ids[k], "label_name": tax.name(k),
                     "support": int(y.sum()), "threshold": float(thr[k]), "tp": tp, "fp": fp, "fn": fn,
                     "precision": prec, "recall": rec, "f1": f1, "average_precision": aps[k]})
    return rows


def format_report(rows: List[Dict]) -> str:
    header = f"{'id':<4}{'class':<44}{'support':>8}{'thr':>6}{'P':>7}{'R':>7}{'F1':>7}{'AP':>7}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(f"{r['label_id']:<4}{r['label_name'][:43]:<44}{r['support']:>8}{r['threshold']:>6.2f}"
                     f"{r['precision']:>7.3f}{r['recall']:>7.3f}{r['f1']:>7.3f}{r['average_precision']:>7.3f}")
    return "\n".join(lines)


def optimize_thresholds(probs: np.ndarray, labels: np.ndarray, strategy: str = "f1",
                        grid: np.ndarray | None = None) -> Tuple[np.ndarray, List[Dict]]:
    """Per-class threshold maximizing ``strategy`` on the given (validation) data.

    Tune on validation, apply to test. Classes with no validation positives keep 0.5.
    """
    from sklearn.metrics import precision_recall_fscore_support

    grid = np.arange(0.05, 1.0, 0.05) if grid is None else grid
    K = labels.shape[1]
    thresholds = np.full(K, 0.5)
    stats: List[Dict] = []
    for k in range(K):
        y, s = labels[:, k].astype(int), probs[:, k]
        if y.sum() == 0:
            stats.append({"label_id": k, "threshold": 0.5, "score": 0.0, "support": 0})
            continue
        best = (-1.0, 0.5, 0.0, 0.0)
        for t in grid:
            p, r, f, _ = precision_recall_fscore_support(y, (s >= t).astype(int), average="binary", zero_division=0)
            score = {"f1": f, "recall": r, "precision": p}.get(strategy, f)
            if score > best[0]:
                best = (score, float(t), p, r)
        thresholds[k] = best[1]
        stats.append({"label_id": k, "threshold": best[1], "score": float(best[0]),
                      "precision": float(best[2]), "recall": float(best[3]), "support": int(y.sum())})
    return thresholds, stats


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray, num_iter: int = 50) -> float:
    """Single temperature T > 0 fit on validation (Guo et al. 2017); logits are divided by T."""
    import torch

    logits_t = torch.as_tensor(val_logits, dtype=torch.float32)
    labels_t = torch.as_tensor(val_labels, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.01, max_iter=num_iter)
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(logits_t / torch.exp(log_t), labels_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).item())
