"""Label taxonomy for the DAppSCAN study.

In the Slither study the class list was fixed (39 detector ids). Here it comes
out of the dataset build: which SWC ids have enough annotated files to learn and
evaluate. The builder writes that list into the dataset's ``.meta.json``, and
every module reads it from there. **The dataset meta is the single source of
truth.** Nothing else hardcodes a class list or a class count.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# SWC Registry titles (https://swcregistry.io). Used for names only.
SWC_TITLES: Dict[str, str] = {
    "100": "Function Default Visibility",
    "101": "Integer Overflow and Underflow",
    "102": "Outdated Compiler Version",
    "103": "Floating Pragma",
    "104": "Unchecked Call Return Value",
    "105": "Unprotected Ether Withdrawal",
    "106": "Unprotected SELFDESTRUCT Instruction",
    "107": "Reentrancy",
    "108": "State Variable Default Visibility",
    "109": "Uninitialized Storage Pointer",
    "110": "Assert Violation",
    "111": "Use of Deprecated Solidity Functions",
    "112": "Delegatecall to Untrusted Callee",
    "113": "DoS with Failed Call",
    "114": "Transaction Order Dependence",
    "115": "Authorization through tx.origin",
    "116": "Block values as a proxy for time",
    "117": "Signature Malleability",
    "118": "Incorrect Constructor Name",
    "119": "Shadowing State Variables",
    "120": "Weak Sources of Randomness from Chain Attributes",
    "121": "Missing Protection against Signature Replay Attacks",
    "122": "Lack of Proper Signature Verification",
    "123": "Requirement Violation",
    "124": "Write to Arbitrary Storage Location",
    "125": "Incorrect Inheritance Order",
    "126": "Insufficient Gas Griefing",
    "127": "Arbitrary Jump with Function Type Variable",
    "128": "DoS With Block Gas Limit",
    "129": "Typographical Error",
    "130": "Right-To-Left-Override control character (U+202E)",
    "131": "Presence of unused variables",
    "132": "Unexpected Ether balance",
    "133": "Hash Collisions With Multiple Variable Length Arguments",
    "134": "Message call with hardcoded gas amount",
    "135": "Code With No Effects",
    "136": "Unencrypted Private Data On-Chain",
}


def swc_title(swc: str) -> str:
    """'SWC-107' or '107' -> 'Reentrancy'."""
    return SWC_TITLES.get(str(swc).upper().replace("SWC-", ""), "unknown")


@dataclass(frozen=True)
class Taxonomy:
    """Ordered class list: index ``i`` is column ``i`` of every label/logit matrix.

    ``eval_idx`` (v3): the classes that are SCORED. Every class is trained, but F1 / AP are
    reported over ``eval_idx`` only -- the classes with enough files to evaluate (the builder's
    ``--eval-min-files``). ``None`` = all classes (v1/v2 tables).
    """

    swc_ids: Tuple[str, ...]  # e.g. ("SWC-135", "SWC-101", ...)
    eval_idx: Optional[Tuple[int, ...]] = None

    @property
    def num_classes(self) -> int:
        return len(self.swc_ids)

    def eval_indices(self) -> List[int]:
        return list(self.eval_idx) if self.eval_idx is not None else list(range(self.num_classes))

    def subset(self, idx: Sequence[int]) -> "Taxonomy":
        return Taxonomy(tuple(self.swc_ids[i] for i in idx))

    def name(self, idx: int) -> str:
        swc = self.swc_ids[idx]
        return f"{swc} {swc_title(swc)}"

    def names(self) -> List[str]:
        return [self.name(i) for i in range(self.num_classes)]

    def index(self, swc: str) -> int:
        return self.swc_ids.index(swc)

    @classmethod
    def from_meta(cls, meta: Dict) -> "Taxonomy":
        classes = sorted(meta["classes"], key=lambda c: c["index"])
        ids = tuple(c["swc"] for c in classes)
        assert [c["index"] for c in classes] == list(range(len(ids))), "class indices must be 0..K-1"
        ev = meta.get("eval_classes")
        if ev is None or len(ev) == len(ids):
            return cls(ids)
        ev = tuple(sorted(int(i) for i in ev))
        assert ev and all(0 <= i < len(ids) for i in ev), "eval_classes must index the class list"
        return cls(ids, ev)


def meta_path_for(parquet_path: str) -> Path:
    p = Path(parquet_path)
    return p.with_name(p.stem + ".meta.json")


def load_meta(parquet_path: str) -> Dict:
    """Read the build metadata: the sidecar JSON if present, else the parquet footer."""
    side = meta_path_for(parquet_path)
    if side.exists():
        return json.loads(side.read_text(encoding="utf-8"))
    import pyarrow.parquet as pq

    md = pq.read_schema(parquet_path).metadata or {}
    raw = md.get(b"dappscan_meta")
    if raw is None:
        raise FileNotFoundError(f"no {side.name} next to {parquet_path} and no embedded metadata; "
                                "was this parquet built by `python -m scvd_dapp build`?")
    return json.loads(raw.decode("utf-8"))


def load_taxonomy(parquet_path: str) -> Taxonomy:
    return Taxonomy.from_meta(load_meta(parquet_path))
