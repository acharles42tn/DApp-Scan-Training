"""Logging setup and a SLURM-friendly progress callback."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(log_dir: str, run_name: str = "train") -> logging.Logger:
    """Configure root logging to both a timestamped file and stdout."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{run_name}_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logger = logging.getLogger("scvd_dapp")
    logger.info("Logging to %s", log_file)
    return logger


def make_progress_callback(logger: logging.Logger, total_steps: int, log_every_n_steps: int = 50):
    """HF ``TrainerCallback`` printing one milestone line every N steps + an epoch summary."""
    from transformers import TrainerCallback

    class CleanProgressCallback(TrainerCallback):
        def __init__(self):
            self.total_steps = max(total_steps, 1)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                lr = logs.get("learning_rate")
                logger.info("Step %d/%d (%.1f%%) | loss %.4f%s", state.global_step, self.total_steps,
                            100.0 * state.global_step / self.total_steps, float(logs["loss"]),
                            f" | lr {float(lr):.2e}" if lr is not None else "")

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not metrics:
                return
            logger.info("-" * 80)
            logger.info("EVAL @ epoch %.1f | val loss %s | file macro-AP %.4f | file F1-macro@0.5 %.4f | "
                        "window F1-micro@0.5 %.4f",
                        state.epoch or 0.0,
                        f"{metrics.get('eval_loss', float('nan')):.4f}",
                        metrics.get("eval_file_macro_ap", float("nan")),
                        metrics.get("eval_file_f1_macro_05", float("nan")),
                        metrics.get("eval_window_f1_micro_05", float("nan")))
            logger.info("-" * 80)

    return CleanProgressCallback()
