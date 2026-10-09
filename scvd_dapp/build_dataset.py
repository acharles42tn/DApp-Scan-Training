"""Build the DAppSCAN-source training table.

    python -m scvd_dapp build --raw data/raw/DAppSCAN --out data/dappscan_v1.parquet

Input is a checkout of github.com/InPlusLab/DAppSCAN at ``PINNED_COMMIT``. Only
``DAppSCAN-source/contracts``, ``DAppSCAN-source/SWCsource`` and
``Audit_and_Repository_link.xlsx`` are read; the audit PDFs and the bytecode half
are not needed.

Output (next to ``--out``):

    dappscan_v1.parquet        one row per unique Solidity file (after cleaning + dedup)
    dappscan_v1.meta.json      class list (the trainer's taxonomy), params, counts, fingerprint
    dappscan_v1.excluded.csv   every annotated file that was left out, and why
    DATA_CARD.md               the human-readable version of the above

Each step below fixes something observed in the raw data (numbers are for the
pinned commit):

1. **Annotation markers are stripped and layout is canonicalised.** DAppSCAN
   writes each label into the source as a comment --
   ``// SWC-107-Reentrancy: L399-422`` -- in all 948 annotated files (1,646
   annotations, 1,647 comments). Training on the raw text would let a model
   read the answer. Annotators sometimes inserted the comment on a new line
   and sometimes typed it over a blank line, so removing it leaves a
   missing/extra blank line exactly at the flagged code; annotated files also
   differ in trailing whitespace and tabs (37% vs 13% of files have trailing
   whitespace). Every file -- annotated or not -- is therefore canonicalised:
   comments removed, tabs -> 4 spaces, trailing whitespace and blank lines
   dropped. Annotated line numbers are remapped onto the canonical text.
   After this, all 70 annotated files that also exist unannotated elsewhere in
   DAppSCAN are byte-identical to that copy.
2. **Tests, mocks and pure interfaces are dropped** unless annotated. No
   annotated file matches the test/mock rule (3,726 unannotated ones do);
   interfaces have no bodies (4 annotated interfaces are kept).
3. **Exact duplicates are merged** (30.8% of files; e.g. Synthetix was audited
   ~23 times). Labels are unioned across copies.
4. **Project-level classes are removed.** SWC-102/103 (compiler version /
   floating pragma) are reported once per project against one file: in the
   projects where the floating pragma is flagged, only ~7.5% of the files that
   also use a floating pragma carry the label. As file labels they are ~92%
   false negatives.
5. **Rare classes are removed**, and files whose only annotations are rare
   classes are excluded (they are known-vulnerable, so they cannot serve as
   negatives either).
6. **Splits are grouped by codebase** (the repository in the xlsx), so audits
   of the same code never straddle train/test, and stratified per class.
   Codebases linked by near-duplicate files are merged while the group stays
   small; the residual near-duplicate overlap is measured, reported, and saved
   (``.neardup.csv``) so evaluation can also report F1 on unexposed files.

Unannotated files mean "no SWC weakness annotated", not "secure": the DAppSCAN
authors only mapped SWC-classifiable findings (non-SWC findings were ~82% of all
audit findings), so negatives are noisy by construction.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import datetime as _dt
import hashlib
import json
import logging
import re
import zipfile
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .taxonomy import swc_title

LOGGER = logging.getLogger("scvd_dapp")

SOURCE_REPO = "https://github.com/InPlusLab/DAppSCAN"
PINNED_COMMIT = "66a56619c44770e05c2db600fa6468115ff0dcd5"  # 2025-03-25, "update fix: SWCsource, contracts"
DATASET_VERSION = "dappscan_v1"

PROJECT_LEVEL_SWC = ("SWC-102", "SWC-103")

# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_text(path: Path) -> str:
    """Decode a source file (utf-8, latin-1 fallback), drop a BOM, CRLF -> LF.

    CRLF -> LF keeps the line count unchanged, so annotated line numbers stay valid.
    """
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    if text.startswith("﻿"):
        text = text[1:]
    return text.replace("\r\n", "\n")


def read_xlsx_rows(path: Path) -> List[List[str]]:
    """Minimal reader for the first sheet of an .xlsx (no openpyxl dependency)."""
    import xml.etree.ElementTree as ET

    m = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

    def col_index(ref: str) -> int:
        letters = re.match(r"[A-Z]+", ref).group(0)
        idx = 0
        for ch in letters:
            idx = idx * 26 + (ord(ch) - 64)
        return idx - 1

    with zipfile.ZipFile(path) as z:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.iter(f"{m}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{m}t")))
        sheets = sorted(n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n))
        root = ET.fromstring(z.read(sheets[0]))
        rows: List[List[str]] = []
        for row in root.iter(f"{m}row"):
            cells: Dict[int, str] = {}
            for c in row.iter(f"{m}c"):
                t = c.get("t")
                v = c.find(f"{m}v")
                if t == "s" and v is not None:
                    val = shared[int(v.text)]
                elif t == "inlineStr":
                    val = "".join(x.text or "" for x in c.iter(f"{m}t"))
                else:
                    val = v.text if v is not None and v.text is not None else ""
                cells[col_index(c.get("r"))] = val
            rows.append([cells.get(i, "") for i in range(max(cells) + 1)] if cells else [])
    return rows


def read_audit_index(xlsx: Path) -> Dict[str, Dict[str, str]]:
    """audit dir name -> {company, project, report, repo} from Audit_and_Repository_link.xlsx."""
    rows = read_xlsx_rows(xlsx)
    header = [h.strip() for h in rows[0]]
    col = {name: header.index(name) for name in
           ("File Name", "Audit Company", "Project Name", "Audit Report Link", "Code Repository")}
    out: Dict[str, Dict[str, str]] = {}
    for r in rows[1:]:
        r = r + [""] * (len(header) - len(r))
        name = r[col["File Name"]].strip()
        if not name:
            continue
        out[name] = {"company": r[col["Audit Company"]].strip(),
                     "project": r[col["Project Name"]].strip(),
                     "report": r[col["Audit Report Link"]].strip(),
                     "repo": r[col["Code Repository"]].strip()}
    return out


def read_annotations(swc_root: Path) -> Dict[str, List[Dict]]:
    """``contracts``-relative .sol path -> list of annotation dicts from SWCsource/**.json."""
    out: Dict[str, List[Dict]] = {}
    prefix = "DAppSCAN-source/contracts/"
    for j in sorted(swc_root.rglob("*.json")):
        d = json.loads(j.read_text(encoding="utf-8"))
        fp = d["filePath"]
        if not fp.startswith(prefix):
            raise ValueError(f"unexpected filePath {fp!r} in {j}")
        out[fp[len(prefix):]] = list(d.get("SWCs") or [])
    return out


# --------------------------------------------------------------------------- #
# Annotation markers and line numbers
# --------------------------------------------------------------------------- #
# Matches the DAppSCAN annotation comment, e.g. "// SWC-107-Reentrancy: L11" or the
# one lowercase variant "//swc-Code With No Effects: L83". Anchored on "//", so an
# ordinary comment that merely mentions "SWC" is left alone.
MARKER_RE = re.compile(r"//\s*SWC-", re.IGNORECASE)
MARKER_ID_RE = re.compile(r"//\s*SWC-(\d+)", re.IGNORECASE)


@dataclass
class Cleaned:
    text: str
    deleted: List[int]                       # annotated-file line numbers that were removed
    markers: List[Tuple[int, Optional[str], bool]]  # (annotated line, "SWC-xxx" or None, own_line)
    n_annot_lines: int
    n_lines: int
    _dset: set = field(default_factory=set, repr=False)

    def __post_init__(self):
        self._dset = set(self.deleted)

    def map_line(self, n: int, side: str = "start") -> int:
        """Annotated-file line number -> canonical-file line number.

        A reference to a removed line (a marker, or a blank line) moves to the
        next surviving line when it starts a range and to the previous surviving
        line when it ends one.
        """
        if self.n_lines == 0:
            return 1
        n = max(1, min(int(n), self.n_annot_lines))
        step = 1 if side == "start" else -1
        m = n
        while m in self._dset and 1 <= m + step <= self.n_annot_lines:
            m += step
        if m in self._dset:  # ran off the end in that direction: go the other way
            m = n
            while m in self._dset and 1 <= m - step <= self.n_annot_lines:
                m -= step
        k = bisect.bisect_left(self.deleted, m)  # removed lines strictly before m
        return max(1, min(m - k, self.n_lines))

    def map_range(self, a: int, b: int) -> Tuple[int, int]:
        s, e = self.map_line(a, "start"), self.map_line(b, "end")
        return (s, max(s, e))


def canonicalize(text: str) -> Cleaned:
    """Strip annotation markers and normalise layout so no trace of annotation remains.

    Annotators sometimes put the ``// SWC-...`` comment on a new line and
    sometimes typed it over an existing blank line, so deleting (or blanking)
    marker lines alone would leave a missing/extra blank line right at the
    flagged code. Layout is therefore normalised for EVERY file, annotated or
    not: tabs -> 4 spaces, trailing whitespace stripped, blank lines removed.
    (Checked: after this, all 70 annotated files that have an unannotated copy
    elsewhere in DAppSCAN are byte-identical to that copy.)
    """
    lines = text.split("\n")
    kept: List[str] = []
    deleted: List[int] = []
    markers: List[Tuple[int, Optional[str], bool]] = []
    for i, line in enumerate(lines, start=1):
        m = MARKER_RE.search(line)
        if m:
            idm = MARKER_ID_RE.search(line)
            markers.append((i, f"SWC-{idm.group(1)}" if idm else None, line[: m.start()].strip() == ""))
            line = line[: m.start()]
        line = line.expandtabs(4).rstrip()
        if not line.strip():
            deleted.append(i)
            continue
        kept.append(line)
    return Cleaned("\n".join(kept), deleted, markers, len(lines), len(kept))


_RANGE_SEP = re.compile(r"[,、，;]")  # , 、 ， ;


def parse_line_field(value) -> Tuple[str, List[Tuple[int, int]]]:
    """Parse DAppSCAN's free-text ``lineNumber``.

    Returns (kind, ranges) with kind one of:
      "ranges" -- ranges in annotated-file numbering ("L21-23", "L5, L9", "L10-12、20-25", "205-253")
      "whole"  -- "all contract" / "All"
      "marker" -- the annotator pasted the marker or code line instead of a number
      "none"   -- empty / N/A
    """
    s = str(value if value is not None else "").strip()
    if not s or s.upper() in {"N/A", "NA", "NONE", "-"}:
        return "none", []
    if re.search(r"swc-", s, re.IGNORECASE):
        return "marker", []
    if re.search(r"\ball\b", s, re.IGNORECASE):
        return "whole", []
    ranges: List[Tuple[int, int]] = []
    for part in _RANGE_SEP.split(s):
        nums = [int(x) for x in re.findall(r"\d+", part)]
        if not nums:
            continue
        a, b = nums[0], nums[-1]
        ranges.append((min(a, b), max(a, b)))
    return ("ranges", ranges) if ranges else ("none", [])


def annotation_spans(ann: Dict, cleaned: Cleaned, swc: str, stats: Counter) -> List[Tuple[int, int]]:
    """Canonical-file line spans for one annotation (never empty)."""
    kind, ranges = parse_line_field(ann.get("lineNumber"))
    stats[f"line_{kind}"] += 1
    whole = (1, max(cleaned.n_lines, 1))
    if kind == "ranges":
        out = []
        for a, b in ranges:
            if b > cleaned.n_annot_lines:
                stats["line_beyond_eof"] += 1
            out.append(cleaned.map_range(a, b))
        return out
    if kind == "whole":
        return [whole]
    # No usable number: fall back to where the annotator put the marker.
    pos = [(ln, own) for ln, mswc, own in cleaned.markers if mswc == swc]
    if pos:
        stats["line_from_marker"] += 1
        # an own-line marker sits above the flagged code; an inline marker is on it
        return [(cleaned.map_line(ln + 1 if own else ln),) * 2 for ln, own in pos]
    stats["line_unknown_whole_file"] += 1
    return [whole]


# --------------------------------------------------------------------------- #
# Scope filters and normalisation
# --------------------------------------------------------------------------- #
_TEST_DIR_RE = re.compile(r"(^|/)(test|tests|testing|mock|mocks|harness|harnesses|echidna|fuzz|fuzzing|certora)/",
                          re.IGNORECASE)
_TEST_FILE_CS_RE = re.compile(r"(^|/)(Test|Mock)[A-Z0-9_][^/]*\.sol$|(Test|Mock|Mocks|Tester|Harness)\.sol$")
_TEST_FILE_CI_RE = re.compile(r"\.t\.sol$|(^|/)[^/]*[_-](test|mock)[^/]*\.sol$", re.IGNORECASE)
_COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)
_DECL_RE = re.compile(r"\b(abstract\s+contract|contract|library|interface)\s+[A-Za-z_$][\w$]*")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def is_test_or_mock(rel_in_audit: str) -> bool:
    return bool(_TEST_DIR_RE.search(rel_in_audit) or _TEST_FILE_CS_RE.search(rel_in_audit)
                or _TEST_FILE_CI_RE.search(rel_in_audit))


def strip_comments(text: str) -> str:
    return _COMMENT_RE.sub(" ", text)


def is_interface_only(code_no_comments: str) -> bool:
    kinds = [m.group(1).split()[-1] for m in _DECL_RE.finditer(code_no_comments)]
    return bool(kinds) and all(k == "interface" for k in kinds)


def normalize_code(text: str) -> str:
    """Comments removed, whitespace collapsed: the identity used for deduplication."""
    return re.sub(r"\s+", " ", strip_comments(text)).strip()


# --------------------------------------------------------------------------- #
# Codebase grouping
# --------------------------------------------------------------------------- #
def codebase_key(repo_link: str, audit_dir: str) -> str:
    """Stable key for 'the same code', from the xlsx Code Repository column."""
    s = repo_link or ""
    m = re.search(r"\b(github|gitlab)\.com/([^/\s]+)/([^/\s#?;,]+)", s, re.IGNORECASE)
    if m:
        repo = re.sub(r"\.git$", "", m.group(3).lower())
        return f"{m.group(1).lower()}:{m.group(2).lower()}/{repo}"
    m = re.search(r"GitHub\s*-\s*([\w.-]+)/([\w.-]+)", s, re.IGNORECASE)
    if m:
        return f"github:{m.group(1).lower()}/{m.group(2).lower()}"
    m = re.search(r"0x[0-9a-fA-F]{40}", s)
    if m:
        return f"addr:{m.group(0).lower()}"
    m = re.search(r"tronscan\.org/#/contract/(\w+)", s)
    if m:
        return f"tron:{m.group(1)}"
    return f"dir:{audit_dir}"


class UnionFind:
    def __init__(self):
        self.parent: Dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


# --------------------------------------------------------------------------- #
# Near-duplicate measurement (MinHash + LSH, numpy only)
# --------------------------------------------------------------------------- #
def shingle_set(normalized: str, k: int = 5) -> np.ndarray:
    toks = _TOKEN_RE.findall(normalized)
    if len(toks) < k:
        grams = [" ".join(toks)] if toks else [""]
    else:
        grams = [" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)]
    return np.unique(np.fromiter((zlib.crc32(g.encode("utf-8")) for g in grams), dtype=np.uint64))


def minhash(shingles: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # multiply-shift hashing mod 2**64 (numpy wraps silently), high 32 bits kept
    with np.errstate(over="ignore"):
        h = (a[:, None] * shingles[None, :] + b[:, None]) >> np.uint64(32)
    return h.min(axis=1)


def jaccard(x: np.ndarray, y: np.ndarray) -> float:
    inter = np.intersect1d(x, y, assume_unique=True).size
    return inter / float(x.size + y.size - inter) if (x.size + y.size) else 1.0


def near_duplicate_pairs(shingles: Sequence[np.ndarray], threshold: float, num_perm: int = 64,
                         bands: int = 16, seed: int = 1) -> List[Tuple[int, int, float]]:
    rng = np.random.RandomState(seed)
    a = (rng.randint(1, 2**31 - 1, size=num_perm).astype(np.uint64) << np.uint64(32)) | \
        rng.randint(1, 2**31 - 1, size=num_perm).astype(np.uint64) | np.uint64(1)
    b = rng.randint(0, 2**31 - 1, size=num_perm).astype(np.uint64)
    sigs = np.stack([minhash(s, a, b) for s in shingles])
    rows = num_perm // bands
    buckets: Dict[Tuple, List[int]] = defaultdict(list)
    for i in range(len(shingles)):
        for bi in range(bands):
            buckets[(bi, sigs[i, bi * rows:(bi + 1) * rows].tobytes())].append(i)
    cand = set()
    for members in buckets.values():
        if 1 < len(members) <= 400:
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    cand.add((members[x], members[y]))
    out = []
    for i, j in cand:
        jac = jaccard(shingles[i], shingles[j])
        if jac >= threshold:
            out.append((i, j, jac))
    return sorted(out)


# --------------------------------------------------------------------------- #
# Fold assignment: grouped + multi-label stratified
# --------------------------------------------------------------------------- #
def assign_folds(group_of_row: np.ndarray, Y: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Assign whole groups to folds so each fold gets ~1/n of every class and of the rows.

    Groups containing the rarest classes are placed first (iterative
    stratification, Sechidis et al. 2011, adapted to groups); unlabeled groups
    then fill folds by row count.
    """
    rng = np.random.RandomState(seed)
    groups = np.unique(group_of_row)
    C = Y.shape[1]
    gY = {g: Y[group_of_row == g].sum(0) for g in groups}
    gN = {g: int((group_of_row == g).sum()) for g in groups}
    total = Y.sum(0).astype(float)
    target_c = np.maximum(total / n_folds, 1e-9)
    target_n = len(group_of_row) / n_folds
    fold_c = np.zeros((n_folds, C))
    fold_n = np.zeros(n_folds)
    jitter = {g: rng.rand() for g in groups}

    def order(g):
        pos = np.nonzero(gY[g])[0]
        rarest = total[pos].min() if len(pos) else np.inf
        return (rarest, -gY[g].sum(), -gN[g], jitter[g])

    fold_of_group = {}
    for g in sorted(groups, key=order):
        if gY[g].sum() > 0:
            present = gY[g] > 0
            need = ((target_c - fold_c) / target_c)[:, present].sum(1)
            score = need + 0.5 * (target_n - fold_n) / target_n
        else:
            score = (target_n - fold_n) / target_n
        best = np.flatnonzero(np.isclose(score, score.max()))
        f = int(best[rng.randint(len(best))])
        fold_of_group[g] = f
        fold_c[f] += gY[g]
        fold_n[f] += gN[g]
    return np.array([fold_of_group[g] for g in group_of_row], dtype=np.int64)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
@dataclass
class FileRec:
    rel: str                      # path relative to DAppSCAN-source/contracts
    audit_dir: str
    text: str                     # cleaned text
    norm_hash: str
    swc: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)  # SWC id -> spans (cleaned lines)


def _map_spans_by_text(src_text: str, dst_text: str, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Map line spans from one copy of a file to a whitespace/comment-variant copy."""
    src = [l.strip() for l in src_text.split("\n")]
    dst = [l.strip() for l in dst_text.split("\n")]
    index: Dict[str, List[int]] = defaultdict(list)
    for i, l in enumerate(dst, start=1):
        if l:
            index[l].append(i)

    def find(n: int) -> Optional[int]:
        key = src[n - 1] if 0 < n <= len(src) else ""
        hits = index.get(key)
        if not hits:
            return None
        guess = n * len(dst) / max(len(src), 1)
        return min(hits, key=lambda h: abs(h - guess))

    out = []
    for a, b in spans:
        a2, b2 = find(a), find(b)
        if a2 is None or b2 is None:
            out.append((1, len(dst)))
        else:
            out.append((min(a2, b2), max(a2, b2)))
    return out


def build(raw_root: Path, out_parquet: Path, min_class_files: int = 20, n_folds: int = 5, seed: int = 42,
          keep_project_level: bool = False, keep_interfaces: bool = False, keep_tests: bool = False,
          near_dup_threshold: float = 0.8, near_dup_group_cap: int = 300,
          default_test_fold: int = 0, default_val_fold: int = 1, eval_min_files: int = 0) -> Dict:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    src_root = raw_root / "DAppSCAN-source"
    contracts = src_root / "contracts"
    if not contracts.is_dir():
        raise FileNotFoundError(f"{contracts} not found -- point --raw at the DAppSCAN checkout")
    commit = _git_head(raw_root)
    if commit and commit != PINNED_COMMIT:
        LOGGER.warning("DAppSCAN checkout is at %s, builder was validated at %s", commit, PINNED_COMMIT)

    audit_index = read_audit_index(raw_root / "Audit_and_Repository_link.xlsx")
    annotations = read_annotations(src_root / "SWCsource")
    stats: Counter = Counter()
    excluded: List[Dict] = []

    # ---- 1. read, strip markers, parse annotations -----------------------------------
    sol_paths = sorted(p for p in contracts.rglob("*.sol") if p.is_file())
    stats["sol_files"] = len(sol_paths)
    recs: List[FileRec] = []
    n_markers = n_instances = 0
    for p in sol_paths:
        rel = p.relative_to(contracts).as_posix()
        audit_dir = rel.split("/", 1)[0]
        rel_in_audit = rel.split("/", 1)[1] if "/" in rel else rel
        cleaned = canonicalize(read_text(p))
        n_markers += len(cleaned.markers)
        anns = annotations.get(rel, [])
        rec = FileRec(rel, audit_dir, cleaned.text, "")
        for ann in anns:
            n_instances += 1
            m = re.match(r"\s*SWC-(\d+)", str(ann.get("category", "")))
            if not m:
                stats["annotation_unparseable_category"] += 1
                continue
            swc = f"SWC-{m.group(1)}"
            rec.swc.setdefault(swc, []).extend(annotation_spans(ann, cleaned, swc, stats))
        if anns and not cleaned.markers:
            stats["annotated_without_marker"] += 1
        if re.search(r"//\s*swc-", cleaned.text, re.IGNORECASE):
            raise AssertionError(f"annotation marker survived cleaning in {rel}")
        # scope
        if not rec.swc:
            if not cleaned.text.strip():
                stats["dropped_empty"] += 1
                continue
            if not keep_tests and is_test_or_mock(rel_in_audit):
                stats["dropped_test_mock"] += 1
                continue
            if not keep_interfaces and is_interface_only(strip_comments(cleaned.text)):
                stats["dropped_interface_only"] += 1
                continue
        recs.append(rec)
    missing = set(annotations) - {p.relative_to(contracts).as_posix() for p in sol_paths}
    if missing:
        raise AssertionError(f"{len(missing)} annotation files point at missing .sol files, e.g. {sorted(missing)[:3]}")
    stats["annotation_files"] = len(annotations)
    stats["annotation_instances"] = n_instances
    stats["markers_stripped"] = n_markers
    LOGGER.info("read %d .sol files; %d annotations in %d files; stripped %d label comments",
                len(sol_paths), n_instances, len(annotations), n_markers)
    if n_markers != n_instances:
        # At the pinned commit there is exactly one extra: an orphan "//swc-Code With No Effects"
        # comment in openzeppelin-Recoverable_Wallet with no JSON entry. It is stripped like the rest.
        LOGGER.info("label comments (%d) != JSON annotations (%d); expected +1 at the pinned commit",
                    n_markers, n_instances)

    # ---- 2. dedup on normalised content -----------------------------------------------
    by_hash: Dict[str, List[FileRec]] = defaultdict(list)
    for r in recs:
        r.norm_hash = hashlib.sha1(normalize_code(r.text).encode("utf-8")).hexdigest()
        by_hash[r.norm_hash].append(r)
    stats["in_scope_files"] = len(recs)
    stats["unique_files"] = len(by_hash)

    merged: List[Tuple[FileRec, List[FileRec]]] = []
    for h in sorted(by_hash):
        copies = by_hash[h]
        rep = min(copies, key=lambda r: (-len(r.swc), r.rel))
        union: Dict[str, List[Tuple[int, int]]] = {k: list(v) for k, v in rep.swc.items()}
        for other in copies:
            if other is rep or not other.swc:
                continue
            for swc, spans in other.swc.items():
                mapped = spans if other.text == rep.text else _map_spans_by_text(other.text, rep.text, spans)
                union.setdefault(swc, []).extend(mapped)
                stats["spans_merged_from_copies"] += len(spans)
        rep = FileRec(rep.rel, rep.audit_dir, rep.text, h, {k: sorted(set(v)) for k, v in union.items()})
        merged.append((rep, copies))

    # ---- 3. class selection ------------------------------------------------------------
    file_count = Counter()
    for rep, _ in merged:
        for swc in rep.swc:
            file_count[swc] += 1
    project_level = [] if keep_project_level else [s for s in PROJECT_LEVEL_SWC if s in file_count]
    candidates = [s for s in file_count if s not in project_level]
    included = sorted([s for s in candidates if file_count[s] >= min_class_files],
                      key=lambda s: (-file_count[s], s))
    rare = sorted([s for s in candidates if file_count[s] < min_class_files], key=lambda s: (-file_count[s], s))
    cls_index = {s: i for i, s in enumerate(included)}
    LOGGER.info("classes kept (>= %d files): %s", min_class_files, ", ".join(included))
    LOGGER.info("project-level classes removed: %s | rare classes removed: %s", project_level, rare)

    rows = []
    for rep, copies in merged:
        own = sorted(rep.swc)
        kept = [s for s in own if s in cls_index]
        rare_here = [s for s in own if s in rare]
        if rare_here and not kept:
            excluded.append({"rel_path": rep.rel, "reason": "only_rare_classes", "swc": ";".join(own)})
            continue
        spans = [{"cls": cls_index[s], "swc": s, "start": int(a), "end": int(b)}
                 for s in kept for (a, b) in rep.swc[s]]
        meta = audit_index.get(rep.audit_dir, {})
        rows.append({
            "content_id": rep.norm_hash[:16],
            "rel_path": rep.rel,
            "audit_dir": rep.audit_dir,
            "audit_company": meta.get("company", ""),
            "project_name": meta.get("project", ""),
            "codebase": codebase_key(meta.get("repo", ""), rep.audit_dir),
            "source_code": rep.text,
            "n_lines": rep.text.count("\n") + 1,
            "n_chars": len(rep.text),
            "n_copies": len(copies),
            "copy_audit_dirs": sorted({c.audit_dir for c in copies}),
            "swc_all": own,
            "labels": sorted(cls_index[s] for s in kept),
            "spans_json": json.dumps(spans, separators=(",", ":")),
        })
    df = pd.DataFrame(rows)
    stats["rows"] = len(df)

    # ---- 4. groups and folds ------------------------------------------------------------
    # A group is a codebase (all audits of one repository). Codebases that share
    # near-duplicate files (forks, re-deployments) are merged too, strongest pair
    # first, but only while the merged group stays <= near_dup_group_cap rows:
    # merging without a cap chains ~74% of all rows into one component through
    # common library variants (SafeMath, ERC20, ...), which makes folds impossible.
    uf = UnionFind()
    size: Dict[str, int] = Counter(df["codebase"])
    for key in size:
        uf.find(key)
    shingles = [shingle_set(normalize_code(t)) for t in df["source_code"]]
    pairs = near_duplicate_pairs(shingles, near_dup_threshold, seed=seed)
    n_merges = 0
    if near_dup_group_cap > 0:
        for i, j, _ in sorted(pairs, key=lambda x: (-x[2], x[0], x[1])):
            a, b = uf.find(df.at[i, "codebase"]), uf.find(df.at[j, "codebase"])
            if a != b and size[a] + size[b] <= near_dup_group_cap:
                uf.union(a, b)
                size[uf.find(a)] = size[a] + size[b]
                n_merges += 1
    stats["codebases"] = len(set(df["codebase"]))
    stats["codebase_merges_near_dup"] = n_merges
    roots = [uf.find(k) for k in df["codebase"]]
    root_ids = {r: i for i, r in enumerate(sorted(set(roots)))}
    df["group_id"] = [root_ids[r] for r in roots]

    K = len(included)
    Y = np.zeros((len(df), K), dtype=np.int64)
    for i, labs in enumerate(df["labels"]):
        Y[i, labs] = 1
    df["fold"] = assign_folds(df["group_id"].to_numpy(), Y, n_folds, seed)
    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))

    # ---- 5. reports ----------------------------------------------------------------------
    per_fold = {s: [int(Y[(df["fold"] == f).to_numpy(), cls_index[s]].sum()) for f in range(n_folds)]
                for s in included}
    fold_rows = [int((df["fold"] == f).sum()) for f in range(n_folds)]
    fold_pos = [int(((df["fold"] == f) & (Y.sum(1) > 0)).sum()) for f in range(n_folds)]
    leakage = _leakage_report(df, pairs, Y, n_folds, default_test_fold, default_val_fold, near_dup_threshold)
    thin = {s: c for s, c in per_fold.items() if min(c) < 3}
    if thin:
        LOGGER.warning("classes with < 3 positives in some fold: %s", thin)

    classes = [{"index": cls_index[s], "swc": s, "title": swc_title(s), "n_files": int(file_count[s]),
                "n_rows": int(Y[:, cls_index[s]].sum()), "per_fold": per_fold[s]} for s in included]
    fingerprint = hashlib.sha256()
    for cid, fold, labs, sp in zip(df["content_id"], df["fold"], df["labels"], df["spans_json"]):
        fingerprint.update(f"{cid}|{fold}|{list(labs)}|{sp}\n".encode())
    meta = {
        "dataset": "DAppSCAN-source",
        "version": out_parquet.stem if out_parquet.stem != DATASET_VERSION else DATASET_VERSION,
        "source_repo": SOURCE_REPO,
        "source_commit": commit or "unknown",
        "pinned_commit": PINNED_COMMIT,
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "builder": "scvd_dapp.build_dataset",
        "params": {"min_class_files": min_class_files, "n_folds": n_folds, "seed": seed,
                   "keep_project_level": keep_project_level, "keep_interfaces": keep_interfaces,
                   "keep_tests": keep_tests, "near_dup_threshold": near_dup_threshold,
                   "near_dup_group_cap": near_dup_group_cap,
                   "canonical_layout": "markers stripped; tabs->4 spaces; trailing whitespace and blank lines removed",
                   "default_test_fold": default_test_fold, "default_val_fold": default_val_fold,
                   "eval_min_files": eval_min_files},
        "classes": classes,
        # v3: classes with >= eval_min_files files are SCORED; every class is trained. Omitted (= all) at 0.
        **({"eval_classes": [cls_index[s] for s in included if file_count[s] >= eval_min_files],
            "eval_min_files": eval_min_files} if eval_min_files > 0 else {}),
        "excluded_classes": {
            "project_level": [{"swc": s, "title": swc_title(s), "n_files": int(file_count[s])} for s in project_level],
            "rare": [{"swc": s, "title": swc_title(s), "n_files": int(file_count[s])} for s in rare],
        },
        "counts": dict(sorted(stats.items())) | {
            "excluded_labeled_files": len(excluded),
            "positive_rows": int((Y.sum(1) > 0).sum()),
            "negative_rows": int((Y.sum(1) == 0).sum()),
            "groups": int(df["group_id"].nunique()),
            "largest_group_rows": int(df["group_id"].value_counts().max()),
        },
        "folds": {"rows": fold_rows, "positive_rows": fold_pos},
        "leakage": leakage,
        "fingerprint": fingerprint.hexdigest(),
    }

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                           b"dappscan_meta": json.dumps(meta).encode("utf-8")})
    pq.write_table(table, out_parquet, compression="zstd")
    stem = out_parquet.with_suffix("")
    Path(str(stem) + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with open(str(stem) + ".excluded.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["rel_path", "reason", "swc"])
        w.writeheader()
        w.writerows(excluded)
    # near-duplicate pairs (row ids), so any split can flag val/test rows exposed to train
    with open(str(stem) + ".neardup.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row_a", "row_b", "jaccard"])
        w.writerows((int(i), int(j), round(float(jac), 4)) for i, j, jac in pairs)
    (out_parquet.parent / "DATA_CARD.md").write_text(_data_card(meta, df, Y, included), encoding="utf-8")
    LOGGER.info("wrote %s (%d rows, %d classes); fingerprint %s", out_parquet, len(df), K, meta["fingerprint"][:12])
    return meta


def _leakage_report(df, pairs, Y, n_folds, test_fold, val_fold, thr) -> Dict:
    folds = df["fold"].to_numpy()
    twin_any = np.zeros(len(df), dtype=bool)
    twin_train_of_test = np.zeros(len(df), dtype=bool)
    train_mask = ~np.isin(folds, [test_fold, val_fold])
    for i, j, _ in pairs:
        if folds[i] != folds[j]:
            twin_any[i] = twin_any[j] = True
        for a, b in ((i, j), (j, i)):
            if folds[a] in (test_fold, val_fold) and train_mask[b]:
                twin_train_of_test[a] = True
    pos = Y.sum(1) > 0

    def frac(mask, sel):
        return round(float(mask[sel].mean()), 4) if sel.any() else 0.0

    t = folds == test_fold
    v = folds == val_fold
    return {
        "near_dup_threshold": thr,
        "near_dup_pairs": len(pairs),
        "cross_fold_pairs": int(sum(1 for i, j, _ in pairs if folds[i] != folds[j])),
        "rows_with_cross_fold_twin": frac(twin_any, np.ones(len(df), bool)),
        "test_rows_with_train_twin": frac(twin_train_of_test, t),
        "test_positive_rows_with_train_twin": frac(twin_train_of_test, t & pos),
        "val_rows_with_train_twin": frac(twin_train_of_test, v),
        "exact_duplicates_across_folds": 0,  # by construction: one row per normalised content
    }


def _data_card(meta: Dict, df, Y, included: List[str]) -> str:
    c = meta["counts"]
    p = meta["params"]
    L = []
    L.append(f"# DAppSCAN training table — `{meta['version']}`\n")
    L.append(f"Built {meta['built_at']} from {meta['source_repo']} at `{meta['source_commit'][:12]}` "
             f"(fingerprint `{meta['fingerprint'][:16]}`).\n")
    L.append("## What a row is\n")
    L.append("One unique Solidity file from `DAppSCAN-source/contracts` after cleaning. `labels` are the "
             "SWC classes below that auditors annotated in the file; `spans_json` gives the annotated line "
             "ranges in the cleaned text (used to label windows). An empty `labels` list means *no SWC weakness "
             "annotated* — not *secure*.\n")
    L.append("## Cleaning\n")
    L.append("DAppSCAN writes every label into the source as a `// SWC-…` comment. These are stripped, and "
             "every file's layout is canonicalised (tabs → 4 spaces, trailing whitespace and blank lines "
             "removed) so no trace of where a comment was remains; annotated line numbers are remapped.\n")
    L.append("| step | count |\n|---|---:|")
    for k, label in [("sol_files", "Solidity files read"),
                     ("annotation_instances", "annotations (SWC instances)"),
                     ("markers_stripped", "`// SWC-…` label comments stripped from the source"),
                     ("dropped_empty", "unannotated empty files dropped"),
                     ("dropped_test_mock", "unannotated test/mock files dropped"),
                     ("dropped_interface_only", "unannotated interface-only files dropped"),
                     ("in_scope_files", "files in scope"),
                     ("unique_files", "unique after merging exact (normalised) duplicates"),
                     ("excluded_labeled_files", "annotated files excluded (only rare classes)"),
                     ("rows", "**rows in the table**"),
                     ("positive_rows", "rows with ≥1 class"),
                     ("negative_rows", "rows with no annotated class")]:
        L.append(f"| {label} | {c.get(k, 0):,} |")
    L.append("")
    L.append("## Classes\n")
    L.append(f"Kept: SWC classes annotated in ≥ {p['min_class_files']} unique files. Folds are grouped by "
             f"codebase ({c.get('codebases', 0)} codebases; {c.get('codebase_merges_near_dup', 0)} merged through "
             f"near-duplicate files, cap {p['near_dup_group_cap']} rows → {c.get('groups', 0)} groups) and "
             f"stratified per class. Default split: test = fold {p['default_test_fold']}, "
             f"val = fold {p['default_val_fold']}, train = the other three.\n")
    L.append("| idx | SWC | title | files | " + " | ".join(f"fold {f}" for f in range(p["n_folds"])) + " |")
    L.append("|---:|---|---|---:|" + "---:|" * p["n_folds"])
    scored = set(meta.get("eval_classes", [cl["index"] for cl in meta["classes"]]))
    for cl in meta["classes"]:
        mark = "" if cl["index"] in scored else " ¹"
        L.append(f"| {cl['index']} | {cl['swc']}{mark} | {cl['title']} | {cl['n_rows']} | "
                 + " | ".join(str(x) for x in cl["per_fold"]) + " |")
    L.append("| | | **rows per fold** | | " + " | ".join(f"{x:,}" for x in meta["folds"]["rows"]) + " |")
    L.append("| | | positive rows per fold | | " + " | ".join(str(x) for x in meta["folds"]["positive_rows"]) + " |")
    L.append("")
    if "eval_classes" in meta:
        L.append(f"¹ Trained but not scored: fewer than {meta['eval_min_files']} files, too few to evaluate. "
                 f"F1 is reported over the other {len(meta['eval_classes'])} classes.\n")
    ex = meta["excluded_classes"]
    if ex["project_level"]:
        L.append("Removed as project-level (reported once per project, not per file): "
                 + ", ".join(f"{e['swc']} {e['title']} ({e['n_files']})" for e in ex["project_level"]) + ".\n")
    if ex["rare"]:
        L.append("Removed as too rare to learn or evaluate: "
                 + ", ".join(f"{e['swc']} ({e['n_files']})" for e in ex["rare"]) + ".\n")
    L.append("## Leakage check\n")
    lk = meta["leakage"]
    L.append(f"Exact duplicates across folds: 0 (one row per normalised file). Near-duplicates "
             f"(Jaccard ≥ {lk['near_dup_threshold']} on 5-token shingles): {lk['near_dup_pairs']:,} pairs, "
             f"{lk['cross_fold_pairs']:,} crossing folds.\n")
    L.append(f"- test rows with a near-twin in train: **{lk['test_rows_with_train_twin']:.1%}** "
             f"(positive test rows: {lk['test_positive_rows_with_train_twin']:.1%})")
    L.append(f"- val rows with a near-twin in train: {lk['val_rows_with_train_twin']:.1%}")
    L.append("")
    L.append("## Annotation line numbers\n")
    L.append("| parsed as | count |\n|---|---:|")
    for k in sorted(k for k in c if k.startswith("line_")):
        L.append(f"| {k[5:]} | {c[k]} |")
    L.append("")
    return "\n".join(L) + "\n"


def _git_head(path: Path) -> Optional[str]:
    head = path / ".git" / "HEAD"
    try:
        ref = head.read_text().strip()
        if ref.startswith("ref:"):
            refpath = path / ".git" / ref.split(" ", 1)[1]
            if refpath.exists():
                return refpath.read_text().strip()
            packed = path / ".git" / "packed-refs"
            for line in packed.read_text().splitlines():
                if line.endswith(ref.split(" ", 1)[1]):
                    return line.split()[0]
            return None
        return ref
    except OSError:
        return None


def add_cli(sub) -> None:
    p = sub.add_parser("build", help="build the DAppSCAN training table from a raw checkout")
    p.add_argument("--raw", required=True, help="path to the DAppSCAN git checkout")
    p.add_argument("--out", default="data/dappscan_v1.parquet")
    p.add_argument("--min-class-files", type=int, default=20)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-project-level", action="store_true", help="keep SWC-102/103 as classes")
    p.add_argument("--keep-interfaces", action="store_true")
    p.add_argument("--keep-tests", action="store_true")
    p.add_argument("--near-dup-threshold", type=float, default=0.8)
    p.add_argument("--eval-min-files", type=int, default=0,
                   help="v3: score only classes with >= this many files (all classes are trained); 0 = all")
    p.add_argument("--near-dup-group-cap", type=int, default=300,
                   help="merge codebases linked by near-duplicate files while the group stays <= this many "
                        "rows (0 = codebase grouping only)")


def run_cli(args: argparse.Namespace) -> None:
    build(Path(args.raw), Path(args.out), min_class_files=args.min_class_files, n_folds=args.n_folds,
          seed=args.seed, keep_project_level=args.keep_project_level, keep_interfaces=args.keep_interfaces,
          keep_tests=args.keep_tests, near_dup_threshold=args.near_dup_threshold,
          near_dup_group_cap=args.near_dup_group_cap, eval_min_files=args.eval_min_files)
