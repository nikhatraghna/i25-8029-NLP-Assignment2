"""BiLSTM POS and NER models for Urdu sequence tagging."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)

from src.crf import LinearChainCRF


class BiLSTMEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        hidden_size=128,
        num_layers=2,
        dropout=0.5,
        bidirectional=True,
        pad_id=0,
        embedding_weights=None,
        freeze_embeddings=False,
    ):
        super().__init__()

        self.pad_id = pad_id
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_id,
        )

        if embedding_weights is not None:
            weights = torch.as_tensor(
                embedding_weights, dtype=torch.float32
            )

            if weights.shape != self.embedding.weight.shape:
                raise ValueError(
                    "Embedding matrix shape does not match "
                    "(vocab_size, embedding_dim)"
                )

            with torch.no_grad():
                self.embedding.weight.copy_(weights)
                self.embedding.weight[pad_id].zero_()

        self.embedding.weight.requires_grad_(
            not freeze_embeddings
        )

        self.input_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_dropout = nn.Dropout(dropout)
        self.output_dim = hidden_size * (
            2 if bidirectional else 1
        )

    def forward(self, input_ids, mask):
        lengths = mask.long().sum(dim=1)

        if (lengths == 0).any():
            raise ValueError("Empty sequences are unsupported")

        embeddings = self.input_dropout(
            self.embedding(input_ids)
        )

        packed = pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        encoded, _ = self.lstm(packed)

        encoded, _ = pad_packed_sequence(
            encoded,
            batch_first=True,
            total_length=input_ids.size(1),
        )

        return self.output_dropout(encoded)


class BiLSTMTagger(nn.Module):
    """BiLSTM with CE for POS or CRF for NER."""

    def __init__(
        self,
        vocab_size,
        num_tags,
        task="pos",
        embedding_dim=100,
        hidden_size=128,
        num_layers=2,
        dropout=0.5,
        bidirectional=True,
        pad_id=0,
        embedding_weights=None,
        freeze_embeddings=False,
        use_crf=True,
    ):
        super().__init__()

        if task not in {"pos", "ner"}:
            raise ValueError("task must be 'pos' or 'ner'")

        self.task = task
        self.num_tags = num_tags
        self.use_crf = (task == "ner") and use_crf

        self.encoder = BiLSTMEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            pad_id=pad_id,
            embedding_weights=embedding_weights,
            freeze_embeddings=freeze_embeddings,
        )

        self.classifier = nn.Linear(
            self.encoder.output_dim,
            num_tags,
        )

        self.crf = (
            LinearChainCRF(num_tags)
            if self.use_crf else None
        )

    def forward(self, input_ids, mask, tags=None):
        mask = mask.bool()
        features = self.encoder(input_ids, mask)
        logits = self.classifier(features)

        output = {"logits": logits}

        if tags is not None:
            if self.use_crf:
                # CRF requires valid IDs even at padded positions.
                safe_tags = tags.masked_fill(~mask, 0)
                output["loss"] = self.crf(
                    logits, safe_tags, mask
                )
            else:
                flat_logits = logits.reshape(-1, self.num_tags)
                flat_tags = tags.reshape(-1)

                token_losses = F.cross_entropy(
                    flat_logits,
                    flat_tags,
                    ignore_index=-100,
                    reduction="none",
                ).reshape_as(tags)

                valid = mask & (tags != -100)
                output["loss"] = (
                    (token_losses * valid).sum()
                    / valid.sum().clamp_min(1)
                )

        return output

    @torch.no_grad()
    def predict(self, input_ids, mask):
        self.eval()
        logits = self.forward(input_ids, mask)["logits"]

        if self.use_crf:
            return self.crf.decode(logits, mask)

        predictions = logits.argmax(dim=-1)
        lengths = mask.long().sum(dim=1).tolist()

        return [
            predictions[i, :length].tolist()
            for i, length in enumerate(lengths)
        ]
