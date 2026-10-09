"""Cut files into model-sized token windows and label each window from annotated lines.

Why windows: annotated DAppSCAN files are long (median ~2.8K tokens). With the
Slither pipeline's head truncation at 1,024 tokens only 39% of annotated line
ranges start inside what the model sees (80% at 4,096). Here every file is tokenized once,
cut into overlapping windows of ``max_length`` tokens (special tokens included),
and each window is labelled positive for a class only if it overlaps a line
range an auditor annotated for that class. At evaluation the file's score for a
class is the max over its windows, so metrics stay at file level.

Tokenization is done on the whole file, then sliced by token index, so every
window's tokens are exactly the tokens the model would see for that stretch of
code in context (no per-line re-tokenization artefacts). Every token carries the
line it starts on (from character offsets, or by counting newlines for slow
tokenizers), which is how windows are matched to annotated line ranges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Tokenizers
# --------------------------------------------------------------------------- #
def special_frame(tokenizer) -> Tuple[List[int], List[int]]:
    """Return (prefix, suffix) special-token ids the tokenizer adds around one sequence."""
    for probe in ("contract", "a", "x = 1"):
        plain = tokenizer(probe, add_special_tokens=False)["input_ids"]
        full = tokenizer(probe, add_special_tokens=True)["input_ids"]
        n = len(plain)
        for i in range(len(full) - n + 1):
            if full[i:i + n] == plain:
                return list(full[:i]), list(full[i + n:])
    raise RuntimeError("could not locate the content tokens inside the tokenizer's special-token frame")


def _line_starts(text: str) -> np.ndarray:
    return np.asarray([0] + [m.end() for m in re.finditer("\n", text)], dtype=np.int64)


class HFWindower:
    """Windows for any HuggingFace tokenizer.

    Each token needs a line number. Fast tokenizers give exact character offsets.
    Slow (pure-Python) tokenizers -- OpenCoder's ``INFLMTokenizer`` is one -- have
    no offsets, so a token's line is counted instead: the number of newlines in
    the decoded tokens before it (decoded once per vocabulary id). That is exact
    except that a token which merges a newline with the next line's indentation
    is counted on the earlier line, so window edges can shift by one line.
    """

    def __init__(self, tokenizer, max_length: int, overlap: float):
        self.tokenizer = tokenizer
        self.prefix, self.suffix = special_frame(tokenizer)
        self.body = max_length - len(self.prefix) - len(self.suffix)
        if self.body < 16:
            raise ValueError(f"max_length={max_length} leaves only {self.body} content tokens")
        self.stride = max(1, self.body - int(round(self.body * overlap)))
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.use_offsets = bool(getattr(tokenizer, "is_fast", False))
        self._newlines: Dict[int, int] = {}

    def tokenize(self, texts: Sequence[str], batch: int = 128):
        """-> list of (token ids, 1-based line number of each token)."""
        out = []
        for i in range(0, len(texts), batch):
            chunk = list(texts[i:i + batch])
            kw = dict(add_special_tokens=False, return_attention_mask=False, truncation=False, verbose=False)
            if self.use_offsets:
                enc = self.tokenizer(chunk, return_offsets_mapping=True, **kw)
                if "offset_mapping" not in enc:   # tokenizer claims fast but gives no offsets
                    self.use_offsets = False
            if not self.use_offsets:
                enc = self.tokenizer(chunk, **kw)
            for j, ids in enumerate(enc["input_ids"]):
                ids = np.asarray(ids, dtype=np.int64)
                if self.use_offsets:
                    starts = np.asarray(enc["offset_mapping"][j], dtype=np.int64).reshape(-1, 2)[:, 0]
                    lines = np.searchsorted(_line_starts(chunk[j]), starts, side="right")
                else:
                    lines = self._lines_by_newline_count(ids)
                out.append((ids, lines.astype(np.int64)))
        return out

    def _lines_by_newline_count(self, ids: np.ndarray) -> np.ndarray:
        for t in set(ids.tolist()) - self._newlines.keys():
            self._newlines[t] = self.tokenizer.decode([t]).count("\n")
        nl = np.fromiter((self._newlines[t] for t in ids.tolist()), dtype=np.int64, count=len(ids))
        return 1 + np.concatenate([[0], np.cumsum(nl)[:-1]]) if len(ids) else np.zeros(0, dtype=np.int64)

    def frame(self, body_ids: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(self.prefix, dtype=np.int64), body_ids,
                               np.asarray(self.suffix, dtype=np.int64)])


_WORD_RE = re.compile(r"\w+|[^\w\s]")


class RegexWindower:
    """Word-level windows for the from-scratch TextCNN (same tokens as scvd's SimpleTokenizer)."""

    def __init__(self, max_length: int, overlap: float, vocab: Dict[str, int] | None = None):
        self.prefix, self.suffix = [], []
        self.body = max_length
        self.stride = max(1, self.body - int(round(self.body * overlap)))
        self.pad_id = 0
        self.vocab = vocab or {"<PAD>": 0, "<UNK>": 1}

    @staticmethod
    def words(text: str) -> List[str]:
        return [m.group().lower() for m in _WORD_RE.finditer(text)]

    def build_vocab(self, texts: Sequence[str], vocab_size: int, min_freq: int) -> None:
        from collections import Counter

        counter: Counter = Counter()
        for t in texts:
            counter.update(self.words(t))
        keep = [w for w, c in counter.most_common() if c >= min_freq][: vocab_size - 2]
        self.vocab = {"<PAD>": 0, "<UNK>": 1, **{w: i for i, w in enumerate(keep, start=2)}}

    def tokenize(self, texts: Sequence[str], batch: int = 0):
        out = []
        for text in texts:
            ms = list(_WORD_RE.finditer(text))
            ids = np.fromiter((self.vocab.get(m.group().lower(), 1) for m in ms), dtype=np.int64, count=len(ms))
            starts = np.fromiter((m.start() for m in ms), dtype=np.int64, count=len(ms))
            out.append((ids, np.searchsorted(_line_starts(text), starts, side="right").astype(np.int64)))
        return out

    def frame(self, body_ids: np.ndarray) -> np.ndarray:
        return body_ids


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #
def window_starts(n_tokens: int, body: int, stride: int) -> List[int]:
    """Start indices so that windows of ``body`` tokens cover [0, n_tokens) completely."""
    if n_tokens <= body:
        return [0]
    starts = list(range(0, n_tokens - body + 1, stride))
    if starts[-1] + body < n_tokens:
        starts.append(n_tokens - body)
    return starts


@dataclass
class WindowSet:
    input_ids: List[np.ndarray]       # framed token ids per window (no padding)
    labels: np.ndarray                # (n_windows, K) float32, window-level labels
    file_idx: np.ndarray              # (n_windows,) row index of the window's file in the split
    lines: np.ndarray                 # (n_windows, 2) first/last line covered (1-based, inclusive)
    file_annotated: np.ndarray        # (n_windows,) bool: the window's FILE has >= 1 class
    n_files: int
    stats: Dict

    def __len__(self) -> int:
        return len(self.input_ids)

    def subset(self, idx: np.ndarray) -> "WindowSet":
        idx = np.asarray(idx, dtype=np.int64)
        return WindowSet([self.input_ids[i] for i in idx], self.labels[idx], self.file_idx[idx],
                         self.lines[idx], self.file_annotated[idx], self.n_files, dict(self.stats))


def build_windows(texts: Sequence[str], spans: Sequence[List[Tuple[int, int, int]]], file_labels: np.ndarray,
                  windower, mode: str, num_classes: int) -> WindowSet:
    """Tokenize ``texts`` and cut them into windows.

    ``spans[i]`` lists (class, first_line, last_line) annotated in file ``i``;
    ``file_labels`` is the (n_files, K) file-level label matrix.
    mode ``window``: all windows, labels from overlapping spans.
    mode ``truncate``: first window only, labelled with the FILE's labels (Slither-pipeline behaviour).
    """
    toks = windower.tokenize(texts)
    ids_out: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    fidx: List[int] = []
    lines: List[Tuple[int, int]] = []
    covered = np.zeros_like(file_labels, dtype=bool)   # class annotated AND inside some window
    n_tokens = np.zeros(len(texts), dtype=np.int64)

    for i, (ids, tok_lines) in enumerate(toks):
        n_tokens[i] = len(ids)
        n_lines = texts[i].count("\n") + 1
        wstarts = window_starts(len(ids), windower.body, windower.stride)
        if mode == "truncate":
            wstarts = wstarts[:1]
        for s in wstarts:
            e = min(s + windower.body, len(ids))
            if e > s:
                l0, l1 = int(tok_lines[s]), int(tok_lines[e - 1])
            else:
                l0 = l1 = 1
            # A token is dated by the line it STARTS on, so a BPE token that merges a newline with
            # the next line ("\n}") is dated one line early. Interior lines are still covered
            # (windows overlap), but the file's first and last lines are pinned explicitly so that
            # every annotated line is inside at least one window.
            if s == 0:
                l0 = 1
            if e == len(ids):
                l1 = max(l1, n_lines)
            y = np.zeros(num_classes, dtype=np.float32)
            for cls, a, b in spans[i]:
                if a <= l1 and b >= l0:
                    y[cls] = 1.0
            covered[i] |= y.astype(bool)
            if mode == "truncate":
                y = file_labels[i].astype(np.float32)
            ids_out.append(windower.frame(ids[s:e]))
            labels.append(y)
            fidx.append(i)
            lines.append((l0, l1))

    file_labels = np.asarray(file_labels, dtype=bool)
    ann = file_labels.sum()
    per_file = np.bincount(np.asarray(fidx), minlength=len(texts))
    stats = {
        "files": len(texts),
        "windows": len(ids_out),
        "windows_per_file_mean": round(float(per_file.mean()), 2) if len(texts) else 0.0,
        "windows_per_file_max": int(per_file.max()) if len(texts) else 0,
        "tokens_per_file_median": int(np.median(n_tokens)) if len(texts) else 0,
        "positive_windows": int((np.asarray(labels).sum(1) > 0).sum()) if labels else 0,
        # share of annotated (file, class) pairs with an annotated line inside a window the model
        # sees: 1.0 by construction in window mode; the truncation loss in truncate mode
        "label_visibility": round(float((covered & file_labels).sum() / ann), 4) if ann else 1.0,
    }
    return WindowSet(ids_out, np.asarray(labels, dtype=np.float32).reshape(-1, num_classes),
                     np.asarray(fidx, dtype=np.int64), np.asarray(lines, dtype=np.int64).reshape(-1, 2),
                     file_labels.any(1)[np.asarray(fidx, dtype=np.int64)] if fidx else np.zeros(0, bool),
                     len(texts), stats)


def aggregate_max(window_logits: np.ndarray, file_idx: np.ndarray, n_files: int) -> np.ndarray:
    """File-level logits = max over the file's windows (per class)."""
    out = np.full((n_files, window_logits.shape[1]), -np.inf, dtype=np.float32)
    np.maximum.at(out, file_idx, window_logits.astype(np.float32))
    if np.isinf(out).any():
        missing = np.flatnonzero(np.isinf(out).all(1))
        raise ValueError(f"{len(missing)} files have no windows (first: {missing[:5]})")
    return out
