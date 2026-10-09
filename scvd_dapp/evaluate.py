"""File-level test report shared by every backend (and the length baseline).

Inputs are FILE-level logits (window logits already max-pooled per file).
Writes ``test_results.json``, ``thresholds.json`` and, optionally,
``logits_val.npz`` / ``logits_test.npz`` to the run's output_dir.

Headline metric, per the project convention: **F1 = macro-averaged F1 over all
classes, per-class decision thresholds tuned on validation.** (v3 tables: over the
taxonomy's eval classes -- the ones with enough files to score; all are trained.) In this dataset
every class has test support by construction (see DATA_CARD.md), so "all
classes" and "supported classes" are the same number -- the ambiguity that
affected the Slither table cannot arise here. The @0.5 block is diagnostic only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .metrics import (fit_temperature, format_report, macro_average_precision, optimize_thresholds,
                      per_class_report, prf_at, sigmoid)
from .taxonomy import Taxonomy

LOGGER = logging.getLogger("scvd_dapp")

F1_DEFINITION = ("macro-averaged F1 over all classes at file level; per-class decision thresholds "
                 "tuned on the validation fold")


def summarize(test_logits: np.ndarray, test_labels: np.ndarray, tax: Taxonomy, output_dir: str,
              optimize: bool = True, strategy: str = "f1", temperature_scaling: bool = False,
              val_logits: Optional[np.ndarray] = None, val_labels: Optional[np.ndarray] = None,
              test_exposed: Optional[np.ndarray] = None, test_row_ids: Optional[np.ndarray] = None,
              val_row_ids: Optional[np.ndarray] = None, save_logits: bool = True,
              extra: Optional[Dict] = None) -> Dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    test_logits = np.asarray(test_logits, dtype=np.float32)
    test_labels = np.asarray(test_labels, dtype=np.float32)
    all_labels = test_labels
    have_val = val_logits is not None and val_labels is not None

    temperature = 1.0
    if temperature_scaling and have_val:
        temperature = fit_temperature(val_logits, val_labels)
        LOGGER.info("Fitted temperature T=%.3f on validation", temperature)
    probs = sigmoid(test_logits / temperature)

    base = prf_at(probs, test_labels, 0.5)
    thresholds = None
    if optimize and have_val:
        vprobs = sigmoid(np.asarray(val_logits, dtype=np.float32) / temperature)
        thresholds, thr_stats = optimize_thresholds(vprobs, val_labels, strategy)
        with open(out / "thresholds.json", "w", encoding="utf-8") as fh:
            json.dump({"strategy": strategy, "temperature": temperature, "thresholds": thresholds.tolist(),
                       "classes": list(tax.swc_ids), "per_class": thr_stats}, fh, indent=2)
    thr = thresholds if thresholds is not None else 0.5

    # v3: every class is trained, but only the taxonomy's eval classes are scored (all of them
    # for v1/v2 tables). Thresholds above are tuned (and saved) for every class.
    ev = tax.eval_indices()
    all_tax = tax
    if len(ev) < tax.num_classes:
        LOGGER.info("Scoring %d of %d classes (classes with enough files to evaluate)", len(ev), tax.num_classes)
        probs, test_labels = probs[:, ev], test_labels[:, ev]
        thr = thr if np.isscalar(thr) else np.asarray(thr)[ev]
        base = prf_at(probs, test_labels, 0.5)
        tax = tax.subset(ev)

    rows = per_class_report(probs, test_labels, thr, tax)
    LOGGER.info("\n%s", format_report(rows))
    f1 = float(np.mean([r["f1"] for r in rows]))
    supported = [r for r in rows if r["support"] > 0]
    f1_supported = float(np.mean([r["f1"] for r in supported])) if supported else 0.0
    macro_ap = macro_average_precision(test_labels, probs)
    tuned = prf_at(probs, test_labels, thr)

    diagnostics = {
        "f1_macro_supported_only": f1_supported,
        "f1_macro_at_0.5": base["f1_macro"],
        "f1_micro_tuned": tuned["f1_micro"],
        "precision_macro_tuned": tuned["precision_macro"],
        "recall_macro_tuned": tuned["recall_macro"],
    }
    if test_exposed is not None and np.any(test_exposed):
        clean = ~np.asarray(test_exposed, dtype=bool)
        rows_c = per_class_report(probs[clean], test_labels[clean], thr, tax)
        sup_c = [r for r in rows_c if r["support"] > 0]
        diagnostics["test_files_exposed_to_train"] = int((~clean).sum())
        diagnostics["f1_unexposed_supported"] = float(np.mean([r["f1"] for r in sup_c])) if sup_c else 0.0
        diagnostics["f1_unexposed_n_supported"] = len(sup_c)

    LOGGER.info("F1 (macro over %d classes, file level, val-tuned thresholds) = %.4f | macro-AP = %.4f | "
                "[@0.5 macro %.4f | tuned micro %.4f]", len(rows), f1, macro_ap, base["f1_macro"],
                tuned["f1_micro"])

    results = {
        "f1": f1,
        "f1_definition": F1_DEFINITION,
        "macro_ap": macro_ap,
        "n_classes": len(rows),
        "n_supported": len(supported),
        "classes": list(tax.swc_ids),
        "trained_classes": list(all_tax.swc_ids),
        "n_test_files": int(len(test_labels)),
        "n_test_positive_files": int((test_labels.sum(1) > 0).sum()),
        "diagnostics": diagnostics,
        "metrics": base,
        "temperature": temperature,
        "used_thresholds": thresholds is not None,
        "per_class": rows,
    }
    if extra:
        results.update(extra)
    with open(out / "test_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    LOGGER.info("Saved %s", out / "test_results.json")

    if save_logits:
        test_labels = all_labels
        np.savez_compressed(out / "logits_test.npz", logits=test_logits, labels=test_labels.astype(np.uint8),
                            row_id=np.asarray(test_row_ids if test_row_ids is not None else []),
                            exposed=np.asarray(test_exposed if test_exposed is not None else [], dtype=np.uint8),
                            classes=np.asarray(all_tax.swc_ids))
        if have_val:
            np.savez_compressed(out / "logits_val.npz", logits=np.asarray(val_logits, dtype=np.float32),
                                labels=np.asarray(val_labels).astype(np.uint8),
                                row_id=np.asarray(val_row_ids if val_row_ids is not None else []),
                                classes=np.asarray(all_tax.swc_ids))
    return results
