"""Core invariants. Run from the DAppSCAN_Training folder:  python tests/test_core.py
(also pytest-compatible). CPU only, a few seconds; the last test needs the built parquet.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scvd_dapp.build_dataset import canonicalize, parse_line_field  # noqa: E402
from scvd_dapp.windows import RegexWindower, aggregate_max, build_windows, window_starts  # noqa: E402


def test_markers_stripped_and_lines_remapped():
    raw = "pragma solidity ^0.8.0;\n\n// SWC-107-Reentrancy: L4-5\nfunction f() {\n  x.call();  // SWC-104-Unchecked: L5\n}\n"
    c = canonicalize(raw)
    assert "SWC" not in c.text.upper()
    assert c.text == "pragma solidity ^0.8.0;\nfunction f() {\n  x.call();\n}"
    # annotated line 4 ("function f()") -> canonical line 2; line 5 -> 3
    assert c.map_range(4, 5) == (2, 3)
    # a range starting on the (deleted) marker line moves to the next surviving line
    assert c.map_range(3, 5) == (2, 3)
    # a range ending on a deleted blank line moves back to the previous surviving line
    assert c.map_range(1, 2) == (1, 1)


def test_layout_canonical_for_every_file():
    a = canonicalize("contract A {\n\tuint x;   \n\n}\n").text
    b = canonicalize("contract A {\n    uint x;\n}").text
    assert a == b  # tabs, trailing whitespace and blank lines cannot tell files apart


def test_parse_line_field_formats():
    assert parse_line_field("L21-23") == ("ranges", [(21, 23)])
    assert parse_line_field("L454- L457") == ("ranges", [(454, 457)])
    assert parse_line_field("L116、145、295") == ("ranges", [(116, 116), (145, 145), (295, 295)])
    assert parse_line_field("L158-165,194-196") == ("ranges", [(158, 165), (194, 196)])
    assert parse_line_field("205-253") == ("ranges", [(205, 253)])
    assert parse_line_field("L995.1009") == ("ranges", [(995, 1009)])
    assert parse_line_field("L3,all contract")[0] == "whole"
    assert parse_line_field("All")[0] == "whole"
    assert parse_line_field("// SWC-135-Code With No Effects")[0] == "marker"
    assert parse_line_field("N/A")[0] == "none"


def test_windows_cover_every_token():
    for n in (0, 1, 50, 100, 101, 1000, 1023, 5000):
        body, stride = 100, 75
        covered = np.zeros(max(n, 1), bool)
        for s in window_starts(n, body, stride):
            covered[s:s + body] = True
        assert covered[:n].all(), n


def test_window_labels_follow_annotated_lines():
    text = "\n".join(f"line{i} x" for i in range(1, 201))  # 200 lines, 2 words each
    w = RegexWindower(max_length=60, overlap=0.0, vocab={"<PAD>": 0, "<UNK>": 1})
    spans = [[(0, 150, 150)]]              # class 0 annotated on line 150 only
    Y = np.array([[1.0, 0.0]])
    ws = build_windows([text], spans, Y, w, "window", 2)
    pos = ws.labels[:, 0] > 0
    assert pos.sum() == 1                  # exactly the window holding line 150
    l0, l1 = ws.lines[pos][0]
    assert l0 <= 150 <= l1
    assert ws.stats["label_visibility"] == 1.0
    # truncate mode: one window, file-level labels, and the annotated line is NOT visible
    wt = build_windows([text], spans, Y, w, "truncate", 2)
    assert len(wt) == 1 and wt.labels[0, 0] == 1.0 and wt.stats["label_visibility"] == 0.0


def test_first_and_last_lines_always_covered():
    # a tokenizer whose final token merges the last line into the previous one ("\n}"), so no
    # token *starts* on the last line; an annotation on that line must still reach a window
    class MergingWindower:
        prefix, suffix, pad_id, body, stride = [], [], 0, 4, 3

        def tokenize(self, texts):
            return [(np.arange(6), np.array([1, 1, 2, 2, 3, 3]))]   # 4 lines, tokens only on 1-3

        def frame(self, ids):
            return ids

    text = "a b\nc d\ne\n}"
    ws = build_windows([text], [[(0, 4, 4)]], np.array([[1.0]]), MergingWindower(), "window", 1)
    assert ws.lines[0][0] == 1 and ws.lines[-1][1] == 4
    assert ws.labels[-1, 0] == 1.0 and ws.stats["label_visibility"] == 1.0


def test_run_stops_if_selected_score_is_not_reproduced():
    from scvd_dapp.train import _check_reproduced

    _check_reproduced("file_macro_ap", 0.2004, {"file_macro_ap": 0.2004})       # identical: passes
    try:
        _check_reproduced("file_macro_ap", 0.2309, {"file_macro_ap": 0.1080})   # misaligned: must stop
    except RuntimeError:
        return
    raise AssertionError("a misaligned prediction order was not caught")


def test_aggregate_max():
    logits = np.array([[0.1, -2.0], [3.0, -1.0], [-5.0, 4.0]], dtype=np.float32)
    out = aggregate_max(logits, np.array([0, 0, 1]), 2)
    assert np.allclose(out, [[3.0, -1.0], [-5.0, 4.0]])


def test_built_table_has_no_label_text():
    pq = Path(__file__).resolve().parent.parent / "data" / "dappscan_v1.parquet"
    if not pq.exists():
        print("  (skipped: data/dappscan_v1.parquet not built)")
        return
    import pandas as pd

    df = pd.read_parquet(pq, columns=["source_code", "labels", "spans_json", "fold", "group_id"])
    assert not df["source_code"].str.contains(r"//\s*swc-", flags=re.IGNORECASE, regex=True).any()
    assert not df["source_code"].str.contains(r"\n\s*\n", regex=True).any()      # no blank lines anywhere
    # every labelled file has spans for exactly its classes, inside the file
    for code, labs, sp in zip(df["source_code"], df["labels"], df["spans_json"]):
        spans = json.loads(sp)
        assert sorted({s["cls"] for s in spans}) == sorted(labs)
        n = code.count("\n") + 1
        assert all(1 <= s["start"] <= s["end"] <= n for s in spans)
    # no codebase group straddles folds
    assert (df.groupby("group_id")["fold"].nunique() == 1).all()


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
