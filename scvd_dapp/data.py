"""Load the DAppSCAN table, split by fold, window it, and resample the TRAIN windows.

Pipeline (every backend):

    load parquet + taxonomy -> split files by fold (grouped, fixed at build time)
      -> window every split -> resample TRAIN windows only -> train -> evaluate on files

Rules carried over from the Slither pipeline, and why:

* **Resampling touches TRAIN only.** Validation and test keep their natural
  distribution, so thresholds tuned on val and F1 on test mean what they say.
  (The Slither pipeline's ``balance_secure`` ran before the split and thinned
  val/test too; that is intentionally not reproduced here.)
* **Oversampling happens after the split** and duplicates train windows only.
* **Splits are precomputed** in the parquet (``fold``), grouped by codebase, so
  every model and every re-run sees the identical files -- no split code to
  reassemble by hand.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .taxonomy import Taxonomy, load_meta, load_taxonomy
from .windows import WindowSet

LOGGER = logging.getLogger("scvd_dapp")


# --------------------------------------------------------------------------- #
# Loading + splitting
# --------------------------------------------------------------------------- #
def load_table(parquet_path: str) -> Tuple[pd.DataFrame, Taxonomy, Dict]:
    meta = load_meta(parquet_path)
    tax = load_taxonomy(parquet_path)
    df = pd.read_parquet(parquet_path)
    LOGGER.info("Loaded %s: %d files, %d classes (%s, fingerprint %s)", parquet_path, len(df),
                tax.num_classes, meta.get("version"), str(meta.get("fingerprint", ""))[:12])
    return df, tax, meta


def split_files(df: pd.DataFrame, test_fold: int, val_fold: int, max_samples: int | None = None,
                seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = sorted(df["fold"].unique())
    for f in (test_fold, val_fold):
        if f not in folds:
            raise ValueError(f"fold {f} not in dataset folds {folds}")
    test = df[df["fold"] == test_fold]
    val = df[df["fold"] == val_fold]
    train = df[~df["fold"].isin([test_fold, val_fold])]
    if max_samples:
        train, val, test = (_cap_files(x, max_samples, seed) for x in (train, val, test))
    train, val, test = (x.reset_index(drop=True) for x in (train, val, test))
    LOGGER.info("Split (files) -- train %d (folds %s) / val %d (fold %d) / test %d (fold %d)",
                len(train), sorted(set(folds) - {test_fold, val_fold}), len(val), val_fold, len(test), test_fold)
    return train, val, test


def _cap_files(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Smoke-test subsample: up to half positives, rest negatives, deterministic."""
    if len(df) <= cap:
        return df
    pos = df[df["labels"].apply(len) > 0]
    neg = df[df["labels"].apply(len) == 0]
    n_pos = min(len(pos), cap // 2)
    return pd.concat([pos.sample(n=n_pos, random_state=seed),
                      neg.sample(n=min(len(neg), cap - n_pos), random_state=seed)]).sort_values("row_id")


def label_matrix(df: pd.DataFrame, num_classes: int) -> np.ndarray:
    Y = np.zeros((len(df), num_classes), dtype=np.float32)
    for i, labs in enumerate(df["labels"]):
        Y[i, list(labs)] = 1.0
    return Y


def file_spans(df: pd.DataFrame) -> List[List[Tuple[int, int, int]]]:
    return [[(s["cls"], s["start"], s["end"]) for s in json.loads(js)] for js in df["spans_json"]]


def log_label_distribution(df: pd.DataFrame, tax: Taxonomy, name: str) -> None:
    Y = label_matrix(df, tax.num_classes)
    LOGGER.info("%s: %d files, %d with >=1 class", name, len(df), int((Y.sum(1) > 0).sum()))
    for k in range(tax.num_classes):
        LOGGER.info("  %2d %-45s %4d", k, tax.name(k)[:45], int(Y[:, k].sum()))


# --------------------------------------------------------------------------- #
# Near-duplicate exposure (for the "unexposed test files" diagnostic)
# --------------------------------------------------------------------------- #
def exposure_flags(parquet_path: str, eval_df: pd.DataFrame, train_df: pd.DataFrame) -> np.ndarray:
    """True for eval files that have a near-duplicate (Jaccard >= build threshold) in train."""
    from pathlib import Path

    p = Path(parquet_path)
    side = p.with_name(p.stem + ".neardup.csv")
    if not side.exists():
        LOGGER.warning("no %s; exposure diagnostic skipped", side.name)
        return np.zeros(len(eval_df), dtype=bool)
    pairs = pd.read_csv(side)
    train_ids = set(train_df["row_id"].tolist())
    exposed = set()
    for a, b in zip(pairs["row_a"], pairs["row_b"]):
        if b in train_ids:
            exposed.add(int(a))
        if a in train_ids:
            exposed.add(int(b))
    return eval_df["row_id"].isin(exposed).to_numpy()


# --------------------------------------------------------------------------- #
# Train-window resampling
# --------------------------------------------------------------------------- #
def resample_train(ws: WindowSet, neg_ratio: float, oversample: bool, threshold: int, target: int,
                   max_factor: float, tax: Taxonomy, seed: int) -> np.ndarray:
    """Indices into ``ws`` for one training epoch's static dataset.

    * every positive window is kept;
    * negative windows from ANNOTATED files are all kept (hard negatives: the
      rest of a flagged file, which teaches localisation);
    * negative windows from UNANNOTATED files are kept with probability ``neg_ratio``;
    * classes with < ``threshold`` positive windows are oversampled towards
      ``target``, never more than ``max_factor`` x their count.
    """
    rng = np.random.RandomState(seed)
    pos = ws.labels.sum(1) > 0
    hard_neg = ~pos & ws.file_annotated
    easy_neg = ~pos & ~ws.file_annotated
    easy_keep = np.flatnonzero(easy_neg)
    if neg_ratio < 1.0:
        n_keep = int(round(len(easy_keep) * neg_ratio))
        easy_keep = np.sort(rng.choice(easy_keep, size=n_keep, replace=False)) if n_keep else easy_keep[:0]
    idx = np.concatenate([np.flatnonzero(pos), np.flatnonzero(hard_neg), easy_keep])
    LOGGER.info("Train windows: %d positive + %d hard-negative (annotated files) + %d/%d easy-negative "
                "(neg_ratio=%.2f)", int(pos.sum()), int(hard_neg.sum()), len(easy_keep), int(easy_neg.sum()),
                neg_ratio)

    if oversample:
        counts = ws.labels[idx].sum(0)
        extra = []
        for k in np.argsort(counts):
            c = int(counts[k])
            if c == 0 or c >= threshold:
                continue
            need = int(min(target - c, (max_factor - 1.0) * c))
            if need <= 0:
                continue
            pool = idx[ws.labels[idx, k] > 0]
            extra.append(rng.choice(pool, size=need, replace=True))
            LOGGER.info("  oversample %-40s %4d -> %4d windows", tax.name(int(k))[:40], c, c + need)
        if extra:
            idx = np.concatenate([idx] + extra)
    rng.shuffle(idx)
    counts = Counter(int(k) for k in np.nonzero(ws.labels[idx])[1])
    LOGGER.info("Train set: %d windows; positives per class: %s", len(idx),
                ", ".join(f"{tax.swc_ids[k]}={counts.get(k, 0)}" for k in range(tax.num_classes)))
    return idx


# --------------------------------------------------------------------------- #
# Torch / HF plumbing
# --------------------------------------------------------------------------- #
def to_hf_dataset(ws: WindowSet, idx: np.ndarray | None = None):
    from datasets import Dataset

    idx = np.arange(len(ws)) if idx is None else idx
    return Dataset.from_dict({
        "input_ids": [ws.input_ids[i].tolist() for i in idx],
        "labels": ws.labels[idx].tolist(),
    })


class PadCollator:
    """Right-pad a batch to its longest window (multiple of 8); build the attention mask."""

    def __init__(self, pad_id: int, multiple: int = 8):
        self.pad_id = int(pad_id)
        self.multiple = multiple

    def __call__(self, features):
        import torch

        n = max(len(f["input_ids"]) for f in features)
        if self.multiple:
            n = ((n + self.multiple - 1) // self.multiple) * self.multiple
        ids = torch.full((len(features), n), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(features), n), dtype=torch.long)
        for i, f in enumerate(features):
            x = torch.as_tensor(f["input_ids"], dtype=torch.long)
            ids[i, : len(x)] = x
            mask[i, : len(x)] = 1
        labels = torch.as_tensor(np.asarray([f["labels"] for f in features], dtype=np.float32))
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


class WindowTorchDataset:
    """Minimal map-style dataset over a WindowSet (for the native training loop)."""

    def __init__(self, ws: WindowSet, idx: np.ndarray | None = None):
        self.ws = ws
        self.idx = np.arange(len(ws)) if idx is None else np.asarray(idx)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return {"input_ids": self.ws.input_ids[j], "labels": self.ws.labels[j]}
