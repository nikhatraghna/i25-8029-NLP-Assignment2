"""Independent BiLSTM baseline for Urdu news classification.

Scratch token embeddings, two-layer bidirectional LSTM,
masked mean/max pooling and five-class classification.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)


class UrduNewsBiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes=5,
        embedding_dim=128,
        hidden_size=128,
        num_layers=2,
        dropout=0.5,
        pad_id=0,
    ):
        super().__init__()

        self.pad_id = pad_id

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_id,
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        representation_dim = 4 * hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(representation_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = input_ids != self.pad_id

        mask = attention_mask.bool()
        lengths = mask.sum(dim=1)

        if (lengths == 0).any():
            raise ValueError("Empty input sequence")

        embedded = self.embedding_dropout(
            self.embedding(input_ids)
        )

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        encoded, _ = self.lstm(packed)

        encoded, _ = pad_packed_sequence(
            encoded,
            batch_first=True,
            total_length=input_ids.shape[1],
        )

        # Masked mean pooling.
        valid = mask.unsqueeze(-1)
        mean_pool = (
            (encoded * valid).sum(dim=1)
            / lengths.unsqueeze(-1).clamp_min(1)
        )

        # Masked max pooling.
        max_input = encoded.masked_fill(
            ~valid,
            torch.finfo(encoded.dtype).min,
        )
        max_pool = max_input.max(dim=1).values

        representation = torch.cat(
            [mean_pool, max_pool],
            dim=-1,
        )

        return self.classifier(representation)
