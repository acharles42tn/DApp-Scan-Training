"""Sanity baseline: how much F1 does file LENGTH alone buy?

Annotated files are much longer than the rest (median ~2.8K vs ~0.85K tokens),
because auditors flag core contracts, not interfaces or helpers. A classifier
that only sees ``log(chars)`` and ``log(lines)`` therefore scores above zero.
Every model should be read against this floor, not against 0.

One class-weighted logistic regression per class, fit on the train folds; the
thresholds are tuned on val and F1 is computed exactly like the models' (same
``evaluate.summarize``).
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from . import data as datamod
from .evaluate import summarize

LOGGER = logging.getLogger("scvd_dapp")


def run_length_baseline(parquet_path: str, output_dir: str, test_fold: int = 0, val_fold: int = 1,
                        seed: int = 42) -> Dict:
    from sklearn.linear_model import LogisticRegression

    df, tax, meta = datamod.load_table(parquet_path)
    train, val, test = datamod.split_files(df, test_fold, val_fold, None, seed)

    def feats(d):
        return np.c_[np.log1p(d["n_chars"].to_numpy()), np.log1p(d["n_lines"].to_numpy())]

    Xtr, Xva, Xte = feats(train), feats(val), feats(test)
    K = tax.num_classes
    Ytr, Yva, Yte = (datamod.label_matrix(x, K) for x in (train, val, test))
    val_logits = np.zeros((len(val), K), dtype=np.float32)
    test_logits = np.zeros((len(test), K), dtype=np.float32)
    for k in range(K):
        if Ytr[:, k].sum() == 0:   # v3: a rare class can be absent from the training folds
            continue
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed).fit(Xtr, Ytr[:, k])
        val_logits[:, k] = clf.decision_function(Xva)
        test_logits[:, k] = clf.decision_function(Xte)
    exposed = datamod.exposure_flags(parquet_path, test, train)
    extra = {"run_name": "length_baseline", "model_name": "logistic regression on log(chars), log(lines)",
             "backend": "baseline",
             "dataset": {"version": meta.get("version"), "fingerprint": meta.get("fingerprint"),
                         "test_fold": test_fold, "val_fold": val_fold,
                         "files": {"train": len(train), "val": len(val), "test": len(test)}}}
    return summarize(test_logits, Yte, tax, output_dir, val_logits=val_logits, val_labels=Yva,
                     test_exposed=exposed, test_row_ids=test["row_id"].to_numpy(),
                     val_row_ids=val["row_id"].to_numpy(), extra=extra)
