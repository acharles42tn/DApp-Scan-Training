"""v3: thin the files with no annotated weakness ("safe" files) in EVERY fold.

The Slither pipeline's ``balance_secure`` kept 30% of the label-free contracts before the
split, so its validation and test sets were balanced too. v3 goes further, as agreed with
the advisor: in each fold, keep every labelled file and a random sample of unlabelled files
of the same size (``safe_ratio`` = unlabelled kept per labelled file; 1.0 -> about half of
every fold, test included, carries a label).

What stays fixed: the folds (grouped by codebase at build time), every labelled file, the
original ``row_id`` (so the near-duplicate sidecar and exposure flags keep working), the
class list and the eval classes. Only unlabelled rows are dropped. The length-only floor
and TF-IDF are re-run on the balanced table, so every score is read against a floor
measured on the same files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np

from .taxonomy import load_meta

LOGGER = logging.getLogger("scvd_dapp")


def balance_table(in_parquet: str, out_parquet: str, safe_ratio: float = 1.0, seed: int = 42) -> Dict:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    if safe_ratio < 0:
        raise ValueError("safe_ratio must be >= 0")
    src, out = Path(in_parquet), Path(out_parquet)
    meta = load_meta(str(src))
    df = pd.read_parquet(src)
    labelled = (df["labels"].apply(len) > 0).to_numpy()
    folds = df["fold"].to_numpy()
    keep = labelled.copy()
    per_fold = []
    for f in sorted(np.unique(folds)):
        n_lab = int((labelled & (folds == f)).sum())
        unl = np.flatnonzero(~labelled & (folds == f))
        n_keep = min(len(unl), int(round(safe_ratio * n_lab)))
        chosen = np.random.RandomState(seed + int(f)).choice(unl, size=n_keep, replace=False)
        keep[chosen] = True
        per_fold.append({"fold": int(f), "labelled": n_lab, "unlabelled_before": int(len(unl)),
                         "unlabelled_kept": int(n_keep)})
        LOGGER.info("fold %d: %d labelled, kept %d of %d unlabelled", f, n_lab, n_keep, len(unl))
    bal = df[keep].reset_index(drop=True)

    K = len(meta["classes"])
    Y = np.zeros((len(bal), K), dtype=np.int64)
    for i, labs in enumerate(bal["labels"]):
        Y[i, list(labs)] = 1
    n_folds = int(meta["params"]["n_folds"])
    fp = hashlib.sha256()
    for cid, fold, labs, sp in zip(bal["content_id"], bal["fold"], bal["labels"], bal["spans_json"]):
        fp.update(f"{cid}|{fold}|{list(labs)}|{sp}\n".encode())
    fp.update(f"safe_ratio={safe_ratio}|seed={seed}".encode())

    new = dict(meta)
    new["version"] = out.stem
    new["balanced_from"] = {"table": src.name, "version": meta.get("version"),
                            "fingerprint": meta.get("fingerprint")}
    new["balance"] = {"safe_ratio": safe_ratio, "seed": seed, "per_fold": per_fold,
                      "rule": "every labelled file kept; per fold, a random sample of unlabelled files "
                              "(safe_ratio x the fold's labelled files)"}
    new["folds"] = {"rows": [int((bal["fold"] == f).sum()) for f in range(n_folds)],
                    "positive_rows": [int(((bal["fold"] == f).to_numpy() & (Y.sum(1) > 0)).sum())
                                      for f in range(n_folds)]}
    new["counts"] = dict(meta["counts"]) | {"rows": int(len(bal)), "positive_rows": int((Y.sum(1) > 0).sum()),
                                            "negative_rows": int((Y.sum(1) == 0).sum()),
                                            "rows_before_balancing": int(len(df))}
    new["fingerprint"] = fp.hexdigest()
    new.pop("leakage", None)  # recomputed per run (exposure flags) from the filtered sidecar below

    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(bal, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                           b"dappscan_meta": json.dumps(new).encode("utf-8")})
    pq.write_table(table, out, compression="zstd")
    Path(str(out.with_suffix("")) + ".meta.json").write_text(json.dumps(new, indent=2), encoding="utf-8")
    side = src.with_name(src.stem + ".neardup.csv")
    if side.exists():
        pairs = pd.read_csv(side)
        ids = set(bal["row_id"].tolist())
        pairs = pairs[pairs["row_a"].isin(ids) & pairs["row_b"].isin(ids)]
        pairs.to_csv(out.with_name(out.stem + ".neardup.csv"), index=False)
    Path(str(out.with_suffix("")) + ".DATA_CARD.md").write_text(_card(new, bal, Y), encoding="utf-8")
    LOGGER.info("wrote %s: %d rows (%d labelled, %d unlabelled) from %d; fingerprint %s", out, len(bal),
                int((Y.sum(1) > 0).sum()), int((Y.sum(1) == 0).sum()), len(df), new["fingerprint"][:12])
    return new


def _card(meta: Dict, bal, Y) -> str:
    n_folds = int(meta["params"]["n_folds"])
    scored = set(meta.get("eval_classes", [c["index"] for c in meta["classes"]]))
    b = meta["balance"]
    L = [f"# DAppSCAN balanced table — `{meta['version']}`\n",
         f"Made from `{meta['balanced_from']['table']}` (fingerprint "
         f"`{str(meta['balanced_from']['fingerprint'])[:16]}`); fingerprint `{meta['fingerprint'][:16]}`.\n",
         "## Balancing\n",
         f"Every labelled file is kept. In each fold, unlabelled files are randomly sampled down to "
         f"{b['safe_ratio']:g} per labelled file (seed {b['seed']}). This applies to train, validation **and "
         "test**, like the Slither pipeline's `balance_secure` (which kept 30%). Scores on this table are "
         "higher than on the natural test sets and must be read against the length-only floor measured on "
         "the same files.\n",
         "| fold | labelled | unlabelled before | unlabelled kept | rows |",
         "|---:|---:|---:|---:|---:|"]
    for pf, rows in zip(b["per_fold"], meta["folds"]["rows"]):
        L.append(f"| {pf['fold']} | {pf['labelled']} | {pf['unlabelled_before']:,} | {pf['unlabelled_kept']} | "
                 f"{rows:,} |")
    L.append(f"| all | {meta['counts']['positive_rows']} | {meta['counts']['rows_before_balancing'] - meta['counts']['positive_rows']:,} "
             f"| {meta['counts']['negative_rows']} | {meta['counts']['rows']:,} |\n")
    L.append("## Classes\n")
    L.append("| idx | SWC | title | files | scored | " + " | ".join(f"fold {f}" for f in range(n_folds)) + " |")
    L.append("|---:|---|---|---:|:---:|" + "---:|" * n_folds)
    for c in meta["classes"]:
        L.append(f"| {c['index']} | {c['swc']} | {c['title']} | {c['n_rows']} | "
                 f"{'yes' if c['index'] in scored else 'no'} | " + " | ".join(str(x) for x in c["per_fold"]) + " |")
    if "eval_classes" in meta:
        L.append(f"\nAll {len(meta['classes'])} classes are trained. F1 is reported over the "
                 f"{len(meta['eval_classes'])} classes with at least {meta.get('eval_min_files')} files.\n")
    return "\n".join(L) + "\n"


def add_cli(sub) -> None:
    p = sub.add_parser("balance", help="v3: keep every labelled file and thin unlabelled files in every fold")
    p.add_argument("--in", dest="in_parquet", required=True)
    p.add_argument("--out", dest="out_parquet", required=True)
    p.add_argument("--safe-ratio", type=float, default=1.0, help="unlabelled files kept per labelled file")
    p.add_argument("--seed", type=int, default=42)


def run_cli(args: argparse.Namespace) -> None:
    balance_table(args.in_parquet, args.out_parquet, args.safe_ratio, args.seed)
