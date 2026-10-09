"""TF-IDF + logistic regression on whole files: a strong simple baseline and v2's blending partner.

In the v1 probes this plain model beat every transformer on fold 0 (F1 0.058 vs 0.031) and
cleared the length-only floor on all five folds (0.057 +- 0.006 vs 0.036 +- 0.009).

Features: code tokens (identifiers, numbers, 1-2 character operators), unigrams + bigrams,
sublinear TF, at most 200k features fit on the TRAIN folds only; plus standardised
log(chars) and log(lines). One class-balanced logistic regression per class; C in
{0.1, 1, 10} is chosen on validation macro-AP. Same folds and the same F1 procedure
(evaluate.summarize) as every other run. CPU, ~1-2 minutes per fold.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from . import data as datamod
from .evaluate import summarize
from .metrics import macro_average_precision, sigmoid

LOGGER = logging.getLogger("scvd_dapp")
TOKEN_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]{1,2}"


def run_tfidf(parquet_path: str, output_dir: str, test_fold: int = 0, val_fold: int = 1, seed: int = 42,
              max_features: int = 200000) -> Dict:
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    df, tax, meta = datamod.load_table(parquet_path)
    train, val, test = datamod.split_files(df, test_fold, val_fold, None, seed)
    K = tax.num_classes
    Ytr, Yva, Yte = (datamod.label_matrix(x, K) for x in (train, val, test))

    vec = TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False, ngram_range=(1, 2), min_df=3,
                          max_features=max_features, sublinear_tf=True, dtype=np.float32)
    Ttr = vec.fit_transform(train["source_code"].tolist())
    Tva = vec.transform(val["source_code"].tolist())
    Tte = vec.transform(test["source_code"].tolist())

    def lens(d):
        return np.c_[np.log1p(d["n_chars"].to_numpy()), np.log1p(d["n_lines"].to_numpy())]

    Ltr, Lva, Lte = lens(train), lens(val), lens(test)
    mu, sd = Ltr.mean(0), Ltr.std(0) + 1e-9
    Xtr = sp.hstack([Ttr, (Ltr - mu) / sd]).tocsr()
    Xva = sp.hstack([Tva, (Lva - mu) / sd]).tocsr()
    Xte = sp.hstack([Tte, (Lte - mu) / sd]).tocsr()
    LOGGER.info("TF-IDF: %d features | train %d / val %d / test %d files", Xtr.shape[1], len(train), len(val),
                len(test))

    best = None
    for C in (0.1, 1.0, 10.0):
        Sv = np.zeros((len(val), K), dtype=np.float32)
        St = np.zeros((len(test), K), dtype=np.float32)
        for k in range(K):
            if Ytr[:, k].sum() == 0:
                continue
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, solver="liblinear",
                                     random_state=seed).fit(Xtr, Ytr[:, k])
            Sv[:, k] = clf.decision_function(Xva)
            St[:, k] = clf.decision_function(Xte)
        ev = tax.eval_indices()  # v3: choose C on the scored classes only
        ap = macro_average_precision(Yva[:, ev], sigmoid(Sv[:, ev]))
        LOGGER.info("  C=%-5g val macro-AP %.4f", C, ap)
        if best is None or ap > best[0]:
            best = (ap, C, Sv, St)
    ap, C, Sv, St = best
    LOGGER.info("TF-IDF: chose C=%g (val macro-AP %.4f)", C, ap)
    exposed = datamod.exposure_flags(parquet_path, test, train)
    extra = {"run_name": f"tfidf_f{test_fold}", "model_name": "TF-IDF (code tokens, 1-2 grams) + logistic regression",
             "backend": "tfidf",
             "dataset": {"version": meta.get("version"), "fingerprint": meta.get("fingerprint"),
                         "test_fold": test_fold, "val_fold": val_fold,
                         "files": {"train": len(train), "val": len(val), "test": len(test)}},
             "tfidf": {"C": C, "features": int(Xtr.shape[1]), "val_macro_ap": round(float(ap), 5)}}
    return summarize(St, Yte, tax, output_dir, val_logits=Sv, val_labels=Yva, test_exposed=exposed,
                     test_row_ids=test["row_id"].to_numpy(), val_row_ids=val["row_id"].to_numpy(), extra=extra)
