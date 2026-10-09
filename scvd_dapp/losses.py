"""Loss functions for long-tailed multi-label classification (same ASL as scvd).

Asymmetric Loss (Ben-Baruch et al. 2020) down-weights easy negatives, which
dominate here: most windows are negative for most classes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        loss = targets * torch.log(xs_pos.clamp(min=self.eps)) + \
            (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = xs_pos * targets + xs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss = loss * torch.pow(1.0 - pt, gamma)
        return -loss.sum(dim=1).mean()


def make_loss(loss_type: str, gamma_neg: float, gamma_pos: float, clip: float) -> nn.Module:
    if loss_type == "asl":
        return AsymmetricLoss(gamma_neg, gamma_pos, clip)
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"loss_type must be 'asl' or 'bce', got {loss_type!r}")


def make_asl_trainer_cls():
    """``Trainer`` subclass optimizing ASL instead of the model's built-in BCE."""
    from transformers import Trainer

    class AsymmetricLossTrainer(Trainer):
        def __init__(self, *args, asl_gamma_neg: float = 4.0, asl_gamma_pos: float = 0.0,
                     asl_clip: float = 0.05, **kwargs):
            super().__init__(*args, **kwargs)
            self._asl = AsymmetricLoss(gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos, clip=asl_clip)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            loss = self._asl(logits, labels)
            return (loss, outputs) if return_outputs else loss

    return AsymmetricLossTrainer
