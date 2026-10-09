"""OpenMythos backend: the recurrent-depth transformer as a multi-label classifier.

Architecture and loading mirror ``comparison_study/pipeline/run_openmythos.py``
(itself copied from ``Open-Mythos/train_classifier.py``): the OpenMythos
backbone with its LM head replaced by identity, masked mean pooling, dropout,
one linear layer.

Initialisation: OpenMythos has no general-purpose pretrained weights, so the
backbone is loaded from the project's own Slither-trained checkpoint
(``mythos_init_ckpt``, default swc_770m_v1/step_00012500_final.pt) and a fresh
head is created for the DAppSCAN classes. **This differs from the HF models,
which start from their public pretrained weights** -- report it next to the
OpenMythos number. Set ``mythos_init_ckpt: ""`` to train from scratch instead.

Runs in the ``openmythos`` conda env (torch 2.11), like the benchmark arm;
``SC.sh`` selects it from the config's backend.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn

LOGGER = logging.getLogger("scvd_dapp")

# Same shapes as run_openmythos.py / train_classifier.py. vocab_size and
# max_seq_len are overridden from the checkpoint's tensors when one is loaded.
MYTHOS_SIZES: Dict[str, Dict] = {
    "200m": dict(vocab_size=200_000, dim=768, n_heads=12, n_kv_heads=4, max_seq_len=2048, max_loop_iters=8,
                 prelude_layers=2, coda_layers=2, n_experts=8, n_shared_experts=1, n_experts_per_tok=2,
                 expert_dim=768, lora_rank=8, attn_type="gqa"),
    "770m": dict(vocab_size=200_000, dim=1536, n_heads=12, n_kv_heads=4, max_seq_len=4096, max_loop_iters=12,
                 prelude_layers=2, coda_layers=2, n_experts=32, n_shared_experts=2, n_experts_per_tok=2,
                 expert_dim=1536, lora_rank=8, attn_type="gqa"),
    # CPU smoke tests only
    "tiny": dict(vocab_size=200_000, dim=64, n_heads=4, n_kv_heads=2, max_seq_len=2048, max_loop_iters=2,
                 prelude_layers=1, coda_layers=1, n_experts=4, n_shared_experts=1, n_experts_per_tok=2,
                 expert_dim=64, lora_rank=4, attn_type="gqa"),
}


def import_open_mythos(src: str):
    try:
        import open_mythos  # installed in the openmythos env
    except ImportError:
        if src and os.path.isdir(src):
            sys.path.insert(0, src)
            import open_mythos
        else:
            raise ImportError("open_mythos is not importable and mythos_src does not exist: "
                              f"{src!r}. Run OpenMythos configs through SC.sh (openmythos env).")
    return open_mythos


class MythosClassifier(nn.Module):
    """Attribute names (backbone / dropout / classifier) match the checkpoint's keys."""

    LM_HEAD_CANDIDATES = ("lm_head", "output_proj", "head", "output")

    def __init__(self, backbone: nn.Module, num_labels: int, hidden_dim: int, dropout: float,
                 n_loops: int, vocab_limit: int, pad_id: int):
        super().__init__()
        self.backbone = backbone
        for name in self.LM_HEAD_CANDIDATES:
            if hasattr(self.backbone, name):
                setattr(self.backbone, name, nn.Identity())
                break
        else:
            raise RuntimeError(f"no LM head found on OpenMythos (tried {self.LM_HEAD_CANDIDATES})")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.n_loops = n_loops
        self.vocab_limit = vocab_limit
        self.pad_id = pad_id if 0 <= pad_id < vocab_limit else 0

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # ids outside the checkpoint's vocabulary (added special tokens) -> pad, masked out
        oor = input_ids >= self.vocab_limit
        if oor.any():
            input_ids = input_ids.masked_fill(oor, self.pad_id)
            attention_mask = attention_mask.masked_fill(oor, 0)
        try:
            hidden = self.backbone(input_ids, n_loops=self.n_loops)
        except TypeError:
            hidden = self.backbone(input_ids)
        if isinstance(hidden, dict):
            hidden = hidden.get("hidden_states", hidden.get("logits"))
        elif isinstance(hidden, tuple):
            hidden = hidden[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(self.dropout(pooled))


def _load_payload(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as err:  # noqa: BLE001 -- the project's own checkpoint also pickles its train cfg
        LOGGER.info("weights_only load refused (%s); loading the project checkpoint with weights_only=False",
                    type(err).__name__)
        return torch.load(path, map_location="cpu", weights_only=False)


def build_mythos(config, num_classes: int, pad_id: int, tokenizer_len: int) -> Tuple[MythosClassifier, Dict]:
    import_open_mythos(config.mythos_src)
    from open_mythos import OpenMythos
    from open_mythos.main import MythosConfig

    sd, train_cfg = None, {}
    if config.mythos_init_ckpt:
        if not os.path.exists(config.mythos_init_ckpt):
            raise FileNotFoundError(f"mythos_init_ckpt not found: {config.mythos_init_ckpt}")
        LOGGER.info("Loading OpenMythos init checkpoint %s", config.mythos_init_ckpt)
        payload = _load_payload(config.mythos_init_ckpt)
        sd = payload.get("model", payload) if isinstance(payload, dict) else payload
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        if isinstance(payload, dict) and isinstance(payload.get("cfg"), dict):
            train_cfg = payload["cfg"]
    size = config.mythos_size or train_cfg.get("size", "770m")
    kwargs = dict(MYTHOS_SIZES[size])
    if sd is not None:
        if "backbone.embed.weight" in sd:
            kwargs["vocab_size"] = int(sd["backbone.embed.weight"].shape[0])
        if "backbone.freqs_cis" in sd:
            kwargs["max_seq_len"] = int(sd["backbone.freqs_cis"].shape[0])
    else:
        kwargs["vocab_size"] = max(kwargs["vocab_size"] if size != "tiny" else 0, tokenizer_len)
    mcfg = MythosConfig(**kwargs)
    model = MythosClassifier(OpenMythos(mcfg), num_classes, mcfg.dim, config.mythos_dropout,
                             config.mythos_n_loops, mcfg.vocab_size, pad_id)
    if sd is not None:
        backbone_sd = {k: v for k, v in sd.items() if k.startswith("backbone.")}
        missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
        missing = [m for m in missing if not m.startswith("classifier.")]
        if missing:
            raise RuntimeError(f"checkpoint lacks {len(missing)} backbone tensors (e.g. {missing[:4]}); "
                               f"is mythos_size={size!r} right for this checkpoint?")
        LOGGER.info("Loaded %d backbone tensors (%d unexpected keys ignored); new %d-class head",
                    len(backbone_sd) - len(unexpected), len(unexpected), num_classes)
        del sd, backbone_sd
    else:
        LOGGER.warning("OpenMythos initialised from scratch (mythos_init_ckpt empty)")
    n_params = sum(p.numel() for p in model.parameters())
    info = {"size": size, "init_ckpt": config.mythos_init_ckpt or None, "vocab_size": mcfg.vocab_size,
            "max_seq_len": mcfg.max_seq_len, "n_loops": config.mythos_n_loops, "params": n_params,
            "init_train_cfg": {k: v for k, v in train_cfg.items() if isinstance(v, (str, int, float, bool))}}
    LOGGER.info("OpenMythos %s: %.1fM params, vocab %d, max_seq_len %d, n_loops %d", size, n_params / 1e6,
                mcfg.vocab_size, mcfg.max_seq_len, config.mythos_n_loops)
    return model, info
