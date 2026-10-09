"""Blend two finished runs on the same fold (v2: a transformer + the TF-IDF model).

* blend logits = w * logits_a + (1 - w) * logits_b, per file and class, with w on a 0.1 grid
  chosen by validation macro-AP (w = 1 is run A alone and w = 0 is run B alone, so on
  validation the blend can only match or beat the better run).
* Raw logits are averaged, not z-scores: both runs are class-balanced models whose logits
  live on the scale the 0.05-0.95 threshold grid was built for, and at w = 1 (or 0) the blend
  reproduces that run's F1 exactly. (A z-score + recalibration variant lost F1 on fold 0
  only because the coarse threshold grid then fell on different cut-offs.)
* The blended logits are evaluated exactly like a single run (evaluate.summarize:
  per-class thresholds tuned on validation, macro F1 over classes).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .evaluate import summarize
from .metrics import macro_average_precision, sigmoid

LOGGER = logging.getLogger("scvd_dapp")


def _load(run_dir: str, split: str) -> Dict[str, np.ndarray]:
    z = np.load(Path(run_dir) / f"logits_{split}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def run_blend(run_a: str, run_b: str, output_dir: str, parquet_path: Optional[str] = None) -> Dict:
    from .config import _default_parquet
    from .taxonomy import load_taxonomy

    va, vb, ta, tb = _load(run_a, "val"), _load(run_b, "val"), _load(run_a, "test"), _load(run_b, "test")
    for name, x, y in (("val", va, vb), ("test", ta, tb)):
        if x["row_id"].shape != y["row_id"].shape or not np.array_equal(x["row_id"], y["row_id"]):
            raise ValueError(f"{name} files differ between {run_a} and {run_b} (different folds or a smoke test)")
        if not np.array_equal(x["labels"], y["labels"]):
            raise ValueError(f"{name} labels differ between {run_a} and {run_b}")
    Yv, Yt = va["labels"].astype(np.float32), ta["labels"].astype(np.float32)
    la_v, lb_v = va["logits"].astype(np.float32), vb["logits"].astype(np.float32)
    la_t, lb_t = ta["logits"].astype(np.float32), tb["logits"].astype(np.float32)

    tax = load_taxonomy(parquet_path or _default_parquet())
    if la_v.shape[1] != tax.num_classes:
        raise ValueError(f"runs have {la_v.shape[1]} classes but the table has {tax.num_classes}; "
                         "pass the run's --parquet-path")
    ev = tax.eval_indices()  # v3: pick w on the scored classes only
    grid = np.round(np.linspace(0.0, 1.0, 11), 2)
    scores = [(macro_average_precision(Yv[:, ev], sigmoid(w * la_v[:, ev] + (1 - w) * lb_v[:, ev])), float(w))
              for w in grid]
    best_ap, w = max(scores, key=lambda t: (round(t[0], 6), -abs(t[1] - 0.5)))
    LOGGER.info("Blend %s + %s: val macro-AP by w = %s -> w = %.1f (%.4f)", Path(run_a).name, Path(run_b).name,
                ", ".join(f"{g:.1f}:{a:.4f}" for a, g in scores), w, best_ap)
    lv = (w * la_v + (1 - w) * lb_v).astype(np.float32)
    lt = (w * la_t + (1 - w) * lb_t).astype(np.float32)

    meta_a = json.loads((Path(run_a) / "test_results.json").read_text(encoding="utf-8"))
    extra = {"run_name": Path(output_dir).name, "model_name": f"blend({Path(run_a).name}, {Path(run_b).name})",
             "backend": "blend", "dataset": meta_a.get("dataset", {}),
             "blend": {"run_a": str(run_a), "run_b": str(run_b), "w_a": w, "val_macro_ap": round(best_ap, 5),
                       "val_macro_ap_by_w": {f"{g:.1f}": round(a, 5) for a, g in scores}}}
    exposed = ta["exposed"].astype(bool) if "exposed" in ta and ta["exposed"].size else None
    return summarize(lt, Yt, tax, output_dir, val_logits=lv, val_labels=Yv, test_exposed=exposed,
                     test_row_ids=ta["row_id"], val_row_ids=va["row_id"], extra=extra)
