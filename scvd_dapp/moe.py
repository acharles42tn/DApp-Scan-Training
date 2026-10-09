"""Mixture-of-Experts over finished single-model runs.

Unlike scvd's MoE (which trained its own copies of the experts inside the MoE
job), the experts here ARE the single-model runs: every run already trained on
the same folds and saved its file-level logits for train/val/test
(``logits_{train,val,test}.npz``). This stage only fits the gate, so it runs on
CPU in minutes and can be re-run with a different expert set for free.

The gating network is scvd's (global top-k gate + per-class gate, 50/50 mix,
KL load-balancing penalty). As in scvd, the gate is fit on the experts' TRAIN
logits (in-sample for the experts, so optimistic), selected on validation
(file-level macro-AP here), and calibrated on validation (temperature + per-class
thresholds) before the single test evaluation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import _default_parquet
from .evaluate import summarize
from .metrics import macro_average_precision, sigmoid
from .taxonomy import Taxonomy

LOGGER = logging.getLogger("scvd_dapp")


@dataclass
class MoEConfig:
    run_name: str = "moe"
    experts: List[str] = field(default_factory=lambda: ["modernbert", "securebert2", "codellama", "openmythos"])
    outputs_root: str = "./outputs"
    output_dir: str = "./outputs/moe"
    log_dir: str = "./logs"
    parquet_path: str = field(default_factory=_default_parquet)
    gating_hidden_dim: int = 512
    gating_epochs: int = 50
    gating_lr: float = 3e-4
    gating_batch_size: int = 64
    top_k: int = 4
    load_balance_weight: float = 0.5
    early_stopping_patience: int = 8
    temperature_scaling: bool = True
    optimize_thresholds: bool = True
    threshold_strategy: str = "f1"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "MoEConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown MoE config keys in {path}: {sorted(unknown)}")
        return cls(**data)


def _gating_network_cls():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class GatingNetwork(nn.Module):
        """Combine expert logits via a global gate + a per-class gate (top-k). Same as scvd."""

        def __init__(self, num_experts: int, num_classes: int, hidden_dim: int = 256, top_k: int = 2):
            super().__init__()
            self.num_experts, self.num_classes, self.top_k = num_experts, num_classes, min(top_k, num_experts)
            in_dim = num_experts * num_classes
            self.gate = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, num_experts))
            self.class_gate = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_experts * num_classes))

        def forward(self, x):
            b = x.shape[0]
            scores = self.gate(x)
            _, topi = torch.topk(scores, self.top_k, dim=-1)
            mask = torch.zeros_like(scores).scatter_(-1, topi, 1.0)
            gate_w = F.softmax(scores * mask + (1 - mask) * (-1e9), dim=-1)
            class_w = F.softmax(self.class_gate(x).view(b, self.num_experts, self.num_classes), dim=1)
            experts = x.view(b, self.num_experts, self.num_classes)
            combined = 0.5 * (experts * gate_w.unsqueeze(-1)).sum(1) + 0.5 * (experts * class_w).sum(1)
            usage = gate_w.mean(dim=0).clamp(min=1e-8)
            lb = F.kl_div(usage.log(), torch.ones_like(usage) / self.num_experts, reduction="batchmean")
            return combined, gate_w, lb

    return GatingNetwork


def _load_expert(root: Path, name: str) -> Dict:
    d = root / name
    need = [d / f"logits_{s}.npz" for s in ("train", "val", "test")] + [d / "test_results.json"]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        raise FileNotFoundError(f"expert '{name}' is not finished (missing {missing})")
    res = json.loads((d / "test_results.json").read_text(encoding="utf-8"))
    out = {"name": name, "fingerprint": res.get("dataset", {}).get("fingerprint"),
           "folds": (res.get("dataset", {}).get("test_fold"), res.get("dataset", {}).get("val_fold")),
           "max_samples": res.get("dataset", {}).get("max_samples")}
    for s in ("train", "val", "test"):
        z = np.load(d / f"logits_{s}.npz", allow_pickle=False)
        out[s] = {"logits": z["logits"].astype(np.float32), "labels": z["labels"].astype(np.float32),
                  "row_id": z["row_id"], "classes": [str(c) for c in z["classes"]]}
    return out


def run_moe(cfg: MoEConfig) -> Dict:
    import torch
    import torch.nn as nn

    from . import data as datamod

    root = Path(cfg.outputs_root)
    experts = [_load_expert(root, n) for n in cfg.experts]
    if len(experts) < 2:
        raise ValueError("an MoE needs at least two experts")
    smoke = [e["name"] for e in experts if e["max_samples"]]
    if smoke:
        raise ValueError(f"{smoke} are --max-samples smoke runs; refusing to build an MoE on them")
    ref = experts[0]
    for e in experts[1:]:
        for s in ("train", "val", "test"):
            if not np.array_equal(e[s]["row_id"], ref[s]["row_id"]):
                raise ValueError(f"expert '{e['name']}' {s} files differ from '{ref['name']}' (split/fold mismatch)")
            if e[s]["classes"] != ref[s]["classes"]:
                raise ValueError(f"expert '{e['name']}' has a different class list")
        if e["fingerprint"] != ref["fingerprint"]:
            raise ValueError(f"expert '{e['name']}' was trained on a different dataset build")
    tax = Taxonomy(tuple(ref["test"]["classes"]))
    K, E = tax.num_classes, len(experts)
    LOGGER.info("MoE over %d experts (%s), %d classes", E, ", ".join(cfg.experts), K)

    def stack(s):
        return np.clip(np.nan_to_num(np.concatenate([e[s]["logits"] for e in experts], axis=1),
                                     nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)

    Xtr, Xva, Xte = stack("train"), stack("val"), stack("test")
    Ytr, Yva, Yte = ref["train"]["labels"], ref["val"]["labels"], ref["test"]["labels"]

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate = _gating_network_cls()(E, K, cfg.gating_hidden_dim, cfg.top_k).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=cfg.gating_lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.gating_epochs)
    bce = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(cfg.seed)
    ds = torch.utils.data.TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr))
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.gating_batch_size, shuffle=True, generator=g)

    def combined(X):
        gate.eval()
        with torch.no_grad():
            out, w, _ = gate(torch.tensor(X, dtype=torch.float32, device=device))
        return out.cpu().numpy(), w.cpu().numpy()

    best, best_state, bad = -1.0, None, 0
    for epoch in range(cfg.gating_epochs):
        gate.train()
        for bx, by in loader:
            out, _, lb = gate(bx.to(device))
            loss = bce(out, by.to(device)) + cfg.load_balance_weight * lb
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            opt.step()
        sched.step()
        ap = macro_average_precision(Yva, sigmoid(combined(Xva)[0]))
        LOGGER.info("gating epoch %02d | val file macro-AP %.4f", epoch + 1, ap)
        if ap > best:
            best, bad = ap, 0
            best_state = {k: v.detach().clone() for k, v in gate.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.early_stopping_patience:
                LOGGER.info("early stopping the gate at epoch %d", epoch + 1)
                break
    gate.load_state_dict(best_state)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_c, _ = combined(Xva)
    test_c, test_w = combined(Xte)
    import pandas as pd

    exposed = datamod.exposure_flags(cfg.parquet_path, pd.DataFrame({"row_id": ref["test"]["row_id"]}),
                                     pd.DataFrame({"row_id": ref["train"]["row_id"]}))
    from safetensors.torch import save_file

    save_file({k: v.detach().cpu().contiguous() for k, v in gate.state_dict().items()},
              str(out_dir / "gating_best.safetensors"))
    extra = {"run_name": cfg.run_name, "model_name": "MoE(" + "+".join(cfg.experts) + ")", "backend": "moe",
             "dataset": {"fingerprint": ref["fingerprint"], "test_fold": ref["folds"][0],
                         "val_fold": ref["folds"][1]},
             "moe": {**asdict(cfg), "best_val_macro_ap": best,
                     "mean_gate_weight_test": dict(zip(cfg.experts, test_w.mean(0).round(4).tolist()))}}
    return summarize(test_c, Yte, tax, str(out_dir), optimize=cfg.optimize_thresholds,
                     strategy=cfg.threshold_strategy, temperature_scaling=cfg.temperature_scaling,
                     val_logits=val_c, val_labels=Yva, test_exposed=exposed,
                     test_row_ids=ref["test"]["row_id"], val_row_ids=ref["val"]["row_id"], extra=extra)
