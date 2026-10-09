"""From-scratch TextCNN baseline (same architecture as scvd.textcnn).

Its word-level tokenizer lives in ``windows.RegexWindower`` so TextCNN gets the
same whole-file windowing as the transformer models.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """Kim-style CNN: embedding -> parallel conv filters -> max-pool -> linear."""

    def __init__(self, vocab_size: int, embed_dim: int, num_filters: int,
                 filter_sizes: List[int], num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, kernel_size=fs) for fs in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.min_len = max(filter_sizes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids.size(1) < self.min_len:  # very short window: pad so every filter fits
            input_ids = F.pad(input_ids, (0, self.min_len - input_ids.size(1)), value=0)
        x = self.embedding(input_ids).permute(0, 2, 1)          # (B, embed, seq)
        feats = []
        for conv in self.convs:
            c = F.relu(conv(x))                                  # (B, filters, L)
            feats.append(F.max_pool1d(c, c.size(2)).squeeze(2))  # (B, filters)
        return self.fc(self.dropout(torch.cat(feats, dim=1)))
