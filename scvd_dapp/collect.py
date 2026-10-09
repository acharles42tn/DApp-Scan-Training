"""Collect every finished run into one results table (``outputs/RESULTS.md``).

One F1 column, per the project convention (macro over all classes, val-tuned
thresholds, file level). The unexposed-files F1 is a leakage diagnostic and is
kept in a separate table so two F1 variants never share one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def _load(outputs: Path) -> List[Dict]:
    runs = []
    for f in sorted(outputs.glob("*/test_results.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        r["_dir"] = f.parent.name
        runs.append(r)
    return runs


def collect(outputs: str = "./outputs") -> str:
    out = Path(outputs)
    runs = _load(out)
    if not runs:
        return f"no test_results.json under {out}"
    runs.sort(key=lambda r: -r.get("f1", 0.0))
    fps = {r.get("dataset", {}).get("fingerprint") for r in runs}
    L = ["# DAppSCAN results", ""]
    if len(fps) > 1:
        L.append(f"**Warning: runs come from {len(fps)} different dataset builds** "
                 f"({', '.join(str(x)[:12] for x in fps)}); they are not comparable.\n")
    L.append("F1 = macro-averaged F1 over all classes, file level, per-class thresholds tuned on validation.\n")
    L.append("| run | model | input | F1 | macro-AP | smoke test? | train h |")
    L.append("|---|---|---|---:|---:|---|---:|")
    for r in runs:
        inp = r.get("input", {})
        if r.get("backend") == "moe":
            inp_s = f"gate over {len(r.get('moe', {}).get('experts', []))} runs"
        elif r.get("backend") == "baseline":
            inp_s = "file length only"
        else:
            inp_s = f"{inp.get('mode', '?')}@{inp.get('max_length', '?')}"
        smoke = "yes" if r.get("dataset", {}).get("max_samples") else ""
        secs = r.get("runtime", {}).get("train_seconds")
        hours = f"{secs / 3600:.1f}" if secs is not None else "–"
        L.append(f"| {r['_dir']} | {r.get('model_name', '')} | {inp_s} | {r.get('f1', 0):.4f} | "
                 f"{r.get('macro_ap', 0):.4f} | {smoke} | {hours} |")
    classes = runs[0].get("classes", [])
    if classes:
        L += ["", "## Per-class F1", "", "| run | " + " | ".join(classes) + " |",
              "|---|" + "---:|" * len(classes)]
        for r in runs:
            by = {p["swc"]: p["f1"] for p in r.get("per_class", []) if "swc" in p}
            L.append(f"| {r['_dir']} | " + " | ".join(f"{by.get(c, 0):.2f}" for c in classes) + " |")
    L += ["", "## Leakage check (diagnostic)", "",
          "Test files with a near-duplicate (Jaccard >= 0.8) in the training folds, and the F1 over the "
          "remaining files (classes with support only).", "",
          "| run | exposed test files | F1 on unexposed files |", "|---|---:|---:|"]
    for r in runs:
        d = r.get("diagnostics", {})
        if "f1_unexposed_supported" in d:
            L.append(f"| {r['_dir']} | {d.get('test_files_exposed_to_train', 0)} | "
                     f"{d['f1_unexposed_supported']:.4f} |")
    text = "\n".join(L) + "\n"
    (out / "RESULTS.md").write_text(text, encoding="utf-8")
    return text


def collect_cv(outputs: str = "./outputs/v2") -> str:
    """v2 cross-validation table: runs named ``<model>_f<fold>`` -> mean +- sd over folds.

    One F1 column (the project convention) and macro-AP; the per-fold F1 values follow."""
    import re

    import numpy as np

    out = Path(outputs)
    groups: Dict[str, Dict[int, Dict]] = {}
    for f in sorted(out.glob("*/test_results.json")):
        m = re.match(r"^(.*)_f(\d+)$", f.parent.name)
        if not m:
            continue
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if r.get("dataset", {}).get("max_samples"):
            continue                                   # smoke tests never enter the table
        groups.setdefault(m.group(1), {})[int(m.group(2))] = r
    if not groups:
        return f"no <model>_f<fold>/test_results.json under {out}"
    rows = []
    for name, folds in groups.items():
        ks = sorted(folds)
        f1 = np.array([folds[k]["f1"] for k in ks])
        ap = np.array([folds[k]["macro_ap"] for k in ks])
        pr = np.array([folds[k].get("diagnostics", {}).get("precision_macro_tuned", np.nan) for k in ks])
        rc = np.array([folds[k].get("diagnostics", {}).get("recall_macro_tuned", np.nan) for k in ks])
        rows.append((name, f1.mean(), f1.std(), ap.mean(), ap.std(), np.mean(pr), np.mean(rc), ks, f1))
    rows.sort(key=lambda r: -r[1])
    L = [f"# DAppSCAN {out.name} results (cross-validated)", "",
         "F1 = macro-averaged F1 over the scored classes, per-class thresholds tuned on validation "
         "(same as v1). Precision and recall are averaged over the same classes, so F1 is not 2PR/(P+R) "
         "of those columns. Each fold: test = fold k, validation = fold k+1, train = the other three.", "",
         "| run | folds | F1 (mean \u00b1 sd) | macro-AP (mean \u00b1 sd) | precision | recall | F1 per fold |",
         "|---|---|---:|---:|---:|---:|---|"]
    for name, m1, s1, m2, s2, p, r, ks, f1 in rows:
        L.append(f"| {name} | {','.join(map(str, ks))} | {m1:.4f} \u00b1 {s1:.4f} | {m2:.4f} \u00b1 {s2:.4f} | "
                 f"{p:.4f} | {r:.4f} | " + " / ".join(f"{x:.3f}" for x in f1) + " |")
    text = "\n".join(L) + "\n"
    (out / f"RESULTS_{out.name}.md").write_text(text, encoding="utf-8")
    return text
