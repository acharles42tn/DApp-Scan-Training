"""Training configuration: one dataclass every model config fills in.

A run is fully described by a YAML file in ``configs/`` loaded onto the defaults
below; CLI flags override the YAML. The resolved config is written to
``<output_dir>/config.json`` -- the provenance record for every number.

Differences from the Slither pipeline (``code/Training_Code/scvd``) are deliberate
and listed in README.md; the model hyperparameters are carried over unchanged.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

SCVD_ROOT = os.environ.get("SCVD_ROOT") or "/work/projects/phd-acharles42/acharles42"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../DAppSCAN_Training

BACKEND_HF_ENCODER = "hf_encoder"    # BERT-style, full fine-tune
BACKEND_HF_DECODER = "hf_decoder"    # causal LM + classification head, usually LoRA
BACKEND_TEXTCNN = "textcnn"          # from-scratch CNN baseline
BACKEND_OPENMYTHOS = "openmythos"    # OpenMythos recurrent-depth classifier (native loop)
VALID_BACKENDS = {BACKEND_HF_ENCODER, BACKEND_HF_DECODER, BACKEND_TEXTCNN, BACKEND_OPENMYTHOS}
VALID_INPUT_MODES = {"window", "truncate"}
VALID_TRAINING_UNITS = {"window", "file"}


def _default_parquet() -> str:
    return os.environ.get("DAPPSCAN_PARQUET") or str(PACKAGE_ROOT / "data" / "dappscan_v1.parquet")


def _default_cache() -> str:
    if os.environ.get("DAPPSCAN_CACHE"):
        return os.environ["DAPPSCAN_CACHE"]
    # Project-wide cache under $SCVD_ROOT (keeps multi-GB model downloads out of $HOME).
    return os.path.join(SCVD_ROOT, ".cache") if os.path.isdir(SCVD_ROOT) else "./.cache"


@dataclass
class TrainConfig:
    # ----- identity -----
    run_name: str = "run"
    model_name: str = "claudios/codebert-base"
    backend: str = BACKEND_HF_ENCODER
    # v3: start the backbone from an already fine-tuned checkpoint instead of the public base
    # model -- e.g. the Slither-trained ``Training_Code/outputs/<model>/final_model``. Encoders:
    # a full HF checkpoint (its classification head is dropped). Decoders: a PEFT LoRA adapter
    # (SEQ_CLS) that is merged into ``model_name`` first. The tokenizer is then read from the
    # same directory unless ``tokenizer_name`` says otherwise.
    backbone_init: Optional[str] = None
    tokenizer_name: Optional[str] = None

    # ----- data -----
    parquet_path: str = field(default_factory=_default_parquet)
    text_field: str = "source_code"
    test_fold: int = 0
    val_fold: int = 1                      # train = every other fold
    max_samples: Optional[int] = None      # smoke tests: cap on FILES per split
    seed: int = 42

    # ----- windows -----
    # "window": every file is cut into overlapping windows of max_length tokens; a window is
    #           positive for a class only if it overlaps an annotated line range; the file's
    #           score is the max over its windows. The whole file is always seen.
    # "truncate": the Slither pipeline's behaviour -- first max_length tokens, file-level labels.
    input_mode: str = "window"
    max_length: int = 1024                 # tokens per window, special tokens included
    window_overlap: float = 0.25           # fraction of a window shared with the next

    # ----- v2: file-level training (training_unit: file) -----
    # "window" (v1): each window carries the labels of the annotated lines it overlaps; a file's
    #                score is the max over its windows.
    # "file"   (v2): one label vector per FILE; the file's windows are encoded and pooled by a
    #                per-class gated attention (Ilse et al. 2018) before the classifier, so the
    #                model is trained on exactly what it is evaluated on. hf_encoder/hf_decoder only.
    training_unit: str = "window"
    freeze_frac: float = 0.0               # bottom share of transformer layers frozen (encoders);
                                           # decoders: LoRA only on the top (1 - freeze_frac) layers
    max_windows_train: int = 16            # windows per file per training step (random subset if more)
    window_batch: int = 16                 # windows per forward pass while training (whole files)
    eval_window_batch: int = 32            # windows per forward pass at evaluation (no grad)
    files_per_step: int = 16               # files per optimizer step (gradient accumulation)
    head_learning_rate: float = 5.0e-4     # attention pooling + classifier (randomly initialised)
    head_dropout: float = 0.1
    att_dim: int = 256
    pos_weight_cap: float = 10.0           # BCE pos_weight per class = neg/pos in the epoch sample, capped
    file_oversample_threshold: int = 50    # classes with fewer labelled TRAIN files are duplicated...
    file_oversample_target: int = 150      # ...towards this many (capped by oversample_max_factor)

    # ----- train-set sampling (val/test are never resampled) -----
    neg_ratio: float = 0.3                 # keep this share of windows from UNANNOTATED files
    oversample: bool = True
    oversample_threshold: int = 100        # classes with fewer positive windows are oversampled...
    oversample_target: int = 300           # ...towards this many...
    oversample_max_factor: float = 5.0     # ...but never duplicated more than this many times

    # ----- quantization / LoRA -----
    load_in_8bit: bool = False
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    # ----- loss -----
    loss_type: str = "asl"                 # "asl" or "bce"
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 0.0
    asl_clip: float = 0.05

    # ----- optimization -----
    num_epochs: int = 5
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    gradient_checkpointing: bool = False

    # ----- TextCNN-only -----
    vocab_size: int = 30000
    min_token_freq: int = 2
    embed_dim: int = 300
    num_filters: int = 256
    filter_sizes: List[int] = field(default_factory=lambda: [3, 4, 5])
    cnn_dropout: float = 0.5

    # ----- OpenMythos-only -----
    mythos_size: str = "770m"
    mythos_init_ckpt: str = os.path.join(
        SCVD_ROOT, "experiments/2026-05_swc_pretrain/checkpoints/swc_770m_v1/step_00012500_final.pt")
    mythos_src: str = os.path.join(SCVD_ROOT, "experiments/2026-05_openmythos/Open-Mythos")
    mythos_n_loops: int = 4                # recurrent depth used for training AND evaluation
    mythos_dropout: float = 0.1

    # ----- evaluation -----
    optimize_thresholds: bool = True       # per-class thresholds tuned on VAL, applied to test
    threshold_strategy: str = "f1"
    temperature_scaling: bool = False
    select_metric: str = "file_macro_ap"   # model selection / early stopping, on VAL files
    save_logits: bool = True               # file-level val/test logits -> .npz (for MoE / ensembles)

    # ----- runtime / paths -----
    resume_from_checkpoint: bool = False
    dry_run: bool = False                  # build windows, report, exit before loading a model
    dataloader_num_workers: int = 2
    output_dir: str = "./outputs/run"
    log_dir: str = "./logs"
    cache_dir: str = field(default_factory=_default_cache)

    def __post_init__(self):
        pq = Path(self.parquet_path)
        if not pq.is_absolute() and not pq.exists() and (PACKAGE_ROOT / pq).exists():
            self.parquet_path = str(PACKAGE_ROOT / pq)   # configs give data/... relative to the package
        if self.backend not in VALID_BACKENDS:
            raise ValueError(f"backend must be one of {sorted(VALID_BACKENDS)}, got {self.backend!r}")
        if self.training_unit not in VALID_TRAINING_UNITS:
            raise ValueError(f"training_unit must be one of {sorted(VALID_TRAINING_UNITS)}, got {self.training_unit!r}")
        if self.training_unit == "file" and self.backend not in (BACKEND_HF_ENCODER, BACKEND_HF_DECODER):
            raise ValueError("training_unit 'file' is implemented for hf_encoder / hf_decoder backends only")
        if self.backbone_init and self.training_unit != "file":
            raise ValueError("backbone_init is implemented for training_unit 'file' only")
        if not 0.0 <= self.freeze_frac < 1.0:
            raise ValueError("freeze_frac must be in [0, 1)")
        if self.input_mode not in VALID_INPUT_MODES:
            raise ValueError(f"input_mode must be one of {sorted(VALID_INPUT_MODES)}, got {self.input_mode!r}")
        if self.test_fold == self.val_fold:
            raise ValueError("test_fold and val_fold must differ")
        if not 0.0 <= self.window_overlap < 1.0:
            raise ValueError("window_overlap must be in [0, 1)")
        for d in (self.output_dir, self.log_dir, self.cache_dir):
            Path(d).mkdir(parents=True, exist_ok=True)

    # ----- IO -----
    @classmethod
    def from_yaml(cls, path: str, overrides: Optional[dict] = None) -> "TrainConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if overrides:
            data.update({k: v for k, v in overrides.items() if v is not None})
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def save(self, path: Optional[str] = None) -> str:
        path = path or str(Path(self.output_dir) / "config.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)
        return path
