"""Skip-gram Word2Vec with Negative Sampling — DS5007."""

import json
import random
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_vocabulary(articles, max_vocab=10000):
    """Build a vocabulary including PAD and UNK."""
    counts = Counter()
    for article in articles.values():
        counts.update(article.split())

    words = [
        w for w, _ in counts.most_common()
        if w not in ("<PAD>", "<UNK>")
    ][:max_vocab - 2]

    vocab = {"<PAD>": 0, "<UNK>": 1}
    vocab.update({w: i + 2 for i, w in enumerate(words)})
    return vocab


def create_training_pairs(articles, vocab, window=5):
    """
    Generate forward and backward centre-context pairs.
    Contexts never cross newline sentence boundaries.
    """
    centres = array("I")
    contexts = array("I")
    unk = vocab["<UNK>"]

    for article_id in sorted(articles):
        for sentence in articles[article_id].splitlines():
            ids = np.asarray(
                [vocab.get(w, unk) for w in sentence.split()],
                dtype=np.uint32
            )

            n = len(ids)
            for offset in range(1, min(window, n - 1) + 1):
                left = ids[:-offset]
                right = ids[offset:]

                centres.extend(left)
                contexts.extend(right)

                centres.extend(right)
                contexts.extend(left)

    return (
        np.frombuffer(centres, dtype=np.uint32).copy(),
        np.frombuffer(contexts, dtype=np.uint32).copy()
    )


def make_noise_distribution(articles, vocab):
    """Compute P(w) proportional to frequency(w)^0.75."""
    counts = np.zeros(len(vocab), dtype=np.float64)
    unk = vocab["<UNK>"]

    for article in articles.values():
        for token in article.split():
            counts[vocab.get(token, unk)] += 1

    # PAD must not be sampled as a negative context.
    counts[vocab["<PAD>"]] = 0

    weights = np.power(counts, 0.75)
    total = weights.sum()

    if total == 0:
        raise ValueError("Noise distribution is empty.")

    probabilities = weights / total
    return torch.tensor(probabilities, dtype=torch.float32)


class SkipGramNS(nn.Module):
    """Independent centre and context embedding matrices."""

    def __init__(self, vocab_size, embedding_dim=100):
        super().__init__()
        self.centre = nn.Embedding(vocab_size, embedding_dim)
        self.context = nn.Embedding(vocab_size, embedding_dim)

        bound = 0.5 / embedding_dim
        nn.init.uniform_(self.centre.weight, -bound, bound)
        nn.init.zeros_(self.context.weight)

    def forward(self, centre_ids, context_ids, negative_ids):
        v = self.centre(centre_ids)
        u_pos = self.context(context_ids)
        u_neg = self.context(negative_ids)

        positive_score = torch.sum(v * u_pos, dim=1)

        negative_score = torch.bmm(
            u_neg, v.unsqueeze(2)
        ).squeeze(2)

        positive_loss = F.logsigmoid(positive_score)
        negative_loss = F.logsigmoid(
            -negative_score
        ).sum(dim=1)

        return -(positive_loss + negative_loss).mean()

    @torch.no_grad()
    def averaged_embeddings(self):
        return (
            (self.centre.weight + self.context.weight) / 2
        ).detach().cpu().numpy()


def train_skipgram(
    articles,
    vocab,
    embedding_dim=100,
    window=5,
    negative_samples=10,
    learning_rate=0.001,
    batch_size=2048,
    epochs=5,
    seed=42,
    log_every=500,
    device=None
):
    """Train Word2Vec using all generated positive pairs."""
    set_seed(seed)

    if device is None:
        device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    centres, contexts = create_training_pairs(
        articles, vocab, window
    )

    pair_count = len(centres)
    if pair_count == 0:
        raise ValueError("No positive training pairs found.")

    print(f"Training pairs: {pair_count:,}")
    print(f"Vocabulary: {len(vocab):,}")
    print(f"Device: {device}")

    noise = make_noise_distribution(
        articles, vocab
    ).to(device)

    model = SkipGramNS(
        len(vocab), embedding_dim
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )

    history = []
    step_history = []
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = np.random.permutation(pair_count)

        total_weighted_loss = 0.0
        seen = 0

        for start in range(0, pair_count, batch_size):
            selected = permutation[
                start:start + batch_size
            ]

            centre_batch = torch.as_tensor(
                centres[selected].astype(np.int64),
                device=device
            )
            context_batch = torch.as_tensor(
                contexts[selected].astype(np.int64),
                device=device
            )

            batch_n = len(selected)

            negatives = torch.multinomial(
                noise,
                batch_n * negative_samples,
                replacement=True
            ).reshape(batch_n, negative_samples)

            # Avoid using the observed positive context
            # as its own negative example.
            collision = negatives.eq(
                context_batch.unsqueeze(1)
            )

            while collision.any().item():
                replacement = torch.multinomial(
                    noise, int(collision.sum().item()),
                    replacement=True
                )
                negatives[collision] = replacement
                collision = negatives.eq(
                    context_batch.unsqueeze(1)
                )

            optimizer.zero_grad(set_to_none=True)

            loss = model(
                centre_batch, context_batch, negatives
            )
            loss.backward()
            optimizer.step()

            total_weighted_loss += (
                loss.item() * batch_n
            )
            seen += batch_n
            global_step += 1

            if global_step % log_every == 0:
                step_history.append({
                    "step": global_step,
                    "loss": float(loss.item())
                })
                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"Step {global_step} | "
                    f"Loss {loss.item():.4f}"
                )

        epoch_loss = total_weighted_loss / seen
        history.append({
            "epoch": epoch,
            "loss": float(epoch_loss)
        })

        print(
            f"Epoch {epoch}/{epochs} completed | "
            f"Mean loss: {epoch_loss:.4f}"
        )

    return model, history, step_history


def save_word2vec(
    model,
    vocab,
    history,
    step_history,
    output_dir,
    model_dir,
    filename="embeddings_w2v.npy"
):
    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    embeddings = model.averaged_embeddings()

    np.save(output_dir / filename, embeddings)

    with open(
        output_dir / "word2idx.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            vocab, f, ensure_ascii=False, indent=2
        )

    with open(
        output_dir / "w2v_training_history.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "epochs": history,
                "steps": step_history
            },
            f,
            indent=2
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": len(vocab),
            "embedding_dim": embeddings.shape[1],
            "history": history
        },
        model_dir / "word2vec_c3.pt"
    )

    return embeddings
