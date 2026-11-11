"""Scratch Transformer encoder for Urdu news topic classification.

No pretrained language model, nn.Transformer, nn.MultiheadAttention,
or nn.TransformerEncoder is used.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model=128, max_length=256):
        super().__init__()

        positions = torch.arange(max_length).float().unsqueeze(1)
        dimensions = torch.arange(0, d_model, 2).float()

        angles = positions * torch.exp(
            -math.log(10000.0) * dimensions / d_model
        )

        pe = torch.zeros(max_length, d_model)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError("Sequence exceeds positional encoding length")
        return x + self.pe[:, :x.size(1)]


class ScratchMultiHeadAttention(nn.Module):
    def __init__(self, d_model=128, num_heads=4, dropout=0.1):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, return_attention=False):
        batch_size, seq_len, d_model = x.shape

        def split_heads(tensor):
            return tensor.reshape(
                batch_size,
                seq_len,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        # Prevent attention to padding tokens.
        key_mask = mask[:, None, None, :]
        scores = scores.masked_fill(
            ~key_mask, torch.finfo(scores.dtype).min
        )

        probabilities = torch.softmax(scores, dim=-1)

        # Keep the original probabilities for visualization.
        attention = probabilities

        probabilities = self.dropout(probabilities)

        context = torch.matmul(probabilities, v)
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(batch_size, seq_len, d_model)

        output = self.out_proj(context)

        # Zero padded query positions.
        output = output * mask.unsqueeze(-1)

        if return_attention:
            return output, attention

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model=128, hidden_dim=512, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """Pre-LN Transformer block."""

    def __init__(
        self,
        d_model=128,
        num_heads=4,
        ff_dim=512,
        dropout=0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = ScratchMultiHeadAttention(
            d_model, num_heads, dropout
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(
            d_model, ff_dim, dropout
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask, return_attention=False):
        if return_attention:
            attended, weights = self.attention(
                self.norm1(x), mask, return_attention=True
            )
        else:
            attended = self.attention(self.norm1(x), mask)
            weights = None

        x = x + self.dropout1(attended)
        x = x + self.dropout2(
            self.feed_forward(self.norm2(x))
        )

        x = x * mask.unsqueeze(-1)

        if return_attention:
            return x, weights

        return x


class UrduNewsTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes=5,
        d_model=128,
        num_heads=4,
        num_layers=4,
        ff_dim=512,
        max_length=256,
        dropout=0.1,
        pad_id=0,
    ):
        super().__init__()

        self.pad_id = pad_id
        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_id,
        )

        self.position_encoding = SinusoidalPositionEncoding(
            d_model, max_length
        )

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, input_ids, attention_mask=None,
                return_attention=False):
        if attention_mask is None:
            attention_mask = input_ids != self.pad_id

        attention_mask = attention_mask.bool()

        if not attention_mask[:, 0].all():
            raise ValueError(
                "First token must be a valid CLS token"
            )

        x = self.embedding(input_ids) * math.sqrt(
            self.embedding.embedding_dim
        )
        x = self.position_encoding(x)
        x = x * attention_mask.unsqueeze(-1)

        last_layer_attention = None

        for index, layer in enumerate(self.layers):
            if return_attention and index == len(self.layers) - 1:
                x, last_layer_attention = layer(
                    x,
                    attention_mask,
                    return_attention=True,
                )
            else:
                x = layer(x, attention_mask)

        x = self.norm(x)

        # CLS token occupies position 0.
        cls_representation = x[:, 0]
        logits = self.classifier(cls_representation)

        if return_attention:
            return logits, last_layer_attention

        return logits
