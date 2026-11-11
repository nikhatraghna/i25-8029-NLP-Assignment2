"""Linear-chain CRF implemented from scratch in PyTorch.

Supports batched negative log-likelihood, padding masks,
and Viterbi decoding. Valid tokens must precede padding.
"""

import torch
import torch.nn as nn


class LinearChainCRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags

        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(
            torch.empty(num_tags, num_tags)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def _validate(self, emissions, mask, tags=None):
        if emissions.ndim != 3:
            raise ValueError("emissions must have shape [B, T, C]")

        B, T, C = emissions.shape
        if C != self.num_tags:
            raise ValueError("Incorrect number of tag logits")

        if mask.shape != (B, T):
            raise ValueError("mask must have shape [B, T]")

        if not mask[:, 0].all():
            raise ValueError("Every sequence must have at least one token")

        if (mask[:, 1:] & ~mask[:, :-1]).any():
            raise ValueError("Mask must contain contiguous valid tokens")

        if tags is not None:
            if tags.shape != (B, T):
                raise ValueError("tags must have shape [B, T]")
            if ((tags < 0) | (tags >= C)).any():
                raise ValueError("Tag IDs out of range")

    def _gold_score(self, emissions, tags, mask):
        B, T, _ = emissions.shape
        arange = torch.arange(B, device=emissions.device)

        score = (
            self.start_transitions[tags[:, 0]]
            + emissions[arange, 0, tags[:, 0]]
        )

        for t in range(1, T):
            step = (
                self.transitions[tags[:, t - 1], tags[:, t]]
                + emissions[arange, t, tags[:, t]]
            )
            score = score + step * mask[:, t]

        lengths = mask.long().sum(dim=1)
        last_tags = tags[arange, lengths - 1]
        score = score + self.end_transitions[last_tags]

        return score

    def _log_partition(self, emissions, mask):
        alpha = self.start_transitions + emissions[:, 0]

        for t in range(1, emissions.size(1)):
            # [B, previous_tag, next_tag]
            scores = (
                alpha.unsqueeze(2)
                + self.transitions.unsqueeze(0)
                + emissions[:, t].unsqueeze(1)
            )

            next_alpha = torch.logsumexp(scores, dim=1)
            alpha = torch.where(
                mask[:, t].unsqueeze(1), next_alpha, alpha
            )

        alpha = alpha + self.end_transitions
        return torch.logsumexp(alpha, dim=1)

    def forward(self, emissions, tags, mask, reduction="mean"):
        mask = mask.bool()
        self._validate(emissions, mask, tags)

        gold = self._gold_score(emissions, tags, mask)
        partition = self._log_partition(emissions, mask)
        nll = partition - gold

        if reduction == "none":
            return nll
        if reduction == "sum":
            return nll.sum()
        if reduction == "mean":
            return nll.mean()
        if reduction == "token_mean":
            return nll.sum() / mask.sum().clamp_min(1)

        raise ValueError(f"Unknown reduction: {reduction}")

    @torch.no_grad()
    def decode(self, emissions, mask):
        mask = mask.bool()
        self._validate(emissions, mask)

        B, T, _ = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history = []

        for t in range(1, T):
            candidates = (
                score.unsqueeze(2)
                + self.transitions.unsqueeze(0)
            )

            best_score, best_prev = candidates.max(dim=1)
            next_score = best_score + emissions[:, t]

            score = torch.where(
                mask[:, t].unsqueeze(1), next_score, score
            )
            history.append(best_prev)

        score = score + self.end_transitions
        best_last = score.argmax(dim=1)

        lengths = mask.long().sum(dim=1).tolist()
        paths = []

        for b, length in enumerate(lengths):
            last = int(best_last[b].item())
            path = [last]

            for t in range(length - 2, -1, -1):
                last = int(history[t][b, last].item())
                path.append(last)

            paths.append(list(reversed(path)))

        return paths
