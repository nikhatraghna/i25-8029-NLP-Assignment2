"""Run and save independent Word2Vec comparison conditions."""

import json
from pathlib import Path

import numpy as np
import torch

from src.word2vec import train_skipgram


def run_condition(
    name,
    articles,
    vocab,
    project_dir,
    embedding_dim=100,
    epochs=5,
    batch_size=2048,
    seed=42,
):
    """Train a Word2Vec condition without overwriting others."""

    project_dir = Path(project_dir)
    embedding_dir = project_dir / "embeddings"
    model_dir = project_dir / "models"

    embedding_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining condition: {name}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Articles: {len(articles)}")

    model, history, steps = train_skipgram(
        articles=articles,
        vocab=vocab,
        embedding_dim=embedding_dim,
        window=5,
        negative_samples=10,
        learning_rate=0.001,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
        log_every=500,
    )

    embeddings = model.averaged_embeddings()

    assert embeddings.shape == (
        len(vocab), embedding_dim
    )
    assert np.isfinite(embeddings).all()

    np.save(
        embedding_dir / f"embeddings_{name}.npy",
        embeddings
    )

    with open(
        embedding_dir / f"word2idx_{name}.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            vocab, f, ensure_ascii=False, indent=2
        )

    with open(
        embedding_dir / f"history_{name}.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {"epochs": history, "steps": steps},
            f, indent=2
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": len(vocab),
            "embedding_dim": embedding_dim,
            "condition": name,
            "history": history,
            "seed": seed,
        },
        model_dir / f"word2vec_{name}.pt"
    )

    print(f"\n{name} completed and saved.")

    return history
