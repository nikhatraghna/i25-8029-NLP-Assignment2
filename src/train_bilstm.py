"""Train and evaluate BiLSTM sequence taggers.

Uses training data only to construct the model vocabulary.
Supports frozen/fine-tuned embeddings, early stopping,
validation macro-F1 and checkpoint saving.
"""

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.bilstm import BiLSTMTagger
from src.sequence_dataset import (
    POS_TAGS,
    NER_TAGS,
    build_token_vocab,
    SequenceTaggingDataset,
    make_collate_fn,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_c3_embeddings(project_dir, vocab, embedding_dim=100):
    """Align pretrained-from-scratch C3 vectors to tagging vocab."""
    project_dir = Path(project_dir)

    vectors = np.load(
        project_dir / "embeddings/embeddings_w2v.npy"
    )
    with open(
        project_dir / "embeddings/word2idx.json",
        encoding="utf-8"
    ) as f:
        w2v_vocab = json.load(f)

    if vectors.ndim != 2 or vectors.shape[1] != embedding_dim:
        raise ValueError(
            f"Expected C3 vectors [N, {embedding_dim}], "
            f"found {vectors.shape}"
        )

    if not isinstance(w2v_vocab, dict):
        raise ValueError("Expected word2idx JSON dictionary")

    matrix = np.random.default_rng(42).normal(
        0.0, 0.05, (len(vocab), embedding_dim)
    ).astype(np.float32)
    matrix[vocab["<PAD>"]] = 0.0

    covered = 0

    for word, index in vocab.items():
        source_id = w2v_vocab.get(word)

        if source_id is None:
            continue

        source_id = int(source_id)
        if not 0 <= source_id < len(vectors):
            raise ValueError(
                f"C3 embedding index out of range: {word}"
            )

        matrix[index] = vectors[source_id]
        covered += 1

    return torch.tensor(matrix), {
        "vocabulary_size": len(vocab),
        "covered_words": covered,
        "coverage_percent": round(
            100 * covered / len(vocab), 2
        ),
        "source_matrix_shape": list(vectors.shape),
    }


def build_loaders(splits, vocab, task, batch_size=16):
    loaders = {}

    for name in ("train", "val", "test"):
        dataset = SequenceTaggingDataset(
            splits[name],
            vocab,
            task=task,
        )

        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            collate_fn=make_collate_fn(task),
        )

    return loaders


def evaluate_tagger(model, loader, device, num_tags):
    model.eval()

    true_ids = []
    predicted_ids = []
    losses = []

    with torch.no_grad():
        for batch in loader:
            x = batch["input_ids"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)

            output = model(x, mask, tags)
            losses.append(float(output["loss"].item()))

            predictions = model.predict(x, mask)

            for i, pred in enumerate(predictions):
                length = int(mask[i].sum().item())
                actual = tags[i, :length].tolist()

                if len(pred) != length:
                    raise ValueError(
                        "Prediction length mismatch"
                    )

                true_ids.extend(actual)
                predicted_ids.extend(pred)

    if not true_ids:
        raise ValueError("Evaluation loader is empty")

    labels = list(range(num_tags))

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(
            accuracy_score(true_ids, predicted_ids)
        ),
        "macro_f1": float(
            f1_score(
                true_ids,
                predicted_ids,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "y_true": true_ids,
        "y_pred": predicted_ids,
    }


def train_model(
    splits,
    project_dir,
    task="pos",
    freeze_embeddings=True,
    use_crf=True,
    bidirectional=True,
    dropout=0.5,
    seed=42,
    epochs=40,
    patience=5,
    batch_size=16,
    learning_rate=1e-3,
    weight_decay=1e-4,
    run_name=None,
):
    if task not in {"pos", "ner"}:
        raise ValueError("task must be pos or ner")

    set_seed(seed)
    project_dir = Path(project_dir)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    for name in ("train", "val", "test"):
        if name not in splits or not splits[name]:
            raise ValueError(f"Missing or empty split: {name}")

    tag_names = POS_TAGS if task == "pos" else NER_TAGS

    vocab = build_token_vocab(splits["train"])
    embedding_weights, coverage = load_c3_embeddings(
        project_dir, vocab
    )

    loaders = build_loaders(
        splits, vocab, task, batch_size
    )

    model = BiLSTMTagger(
        vocab_size=len(vocab),
        num_tags=len(tag_names),
        task=task,
        embedding_dim=100,
        hidden_size=128,
        num_layers=2,
        dropout=dropout,
        bidirectional=bidirectional,
        embedding_weights=embedding_weights,
        freeze_embeddings=freeze_embeddings,
        use_crf=use_crf,
    ).to(device)

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    if run_name is None:
        embedding_mode = (
            "frozen" if freeze_embeddings else "finetuned"
        )
        run_name = f"{task}_{embedding_mode}_seed{seed}"

    checkpoints = project_dir / "models"
    metrics_dir = project_dir / "results/metrics"
    checkpoints.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    stale_epochs = 0

    print("Run:", run_name)
    print("Device:", device)
    print("Embedding coverage:", coverage)
    print("Train/val/test sizes:", [
        len(splits[k]) for k in ("train", "val", "test")
    ])

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for batch in loaders["train"]:
            x = batch["input_ids"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = model(x, mask, tags)["loss"]

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite training loss"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()
            train_losses.append(loss.item())

        val = evaluate_tagger(
            model, loaders["val"], device, len(tag_names)
        )

        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d} | "
            f"train={record['train_loss']:.4f} | "
            f"val_f1={record['val_macro_f1']:.4f} | "
            f"val_acc={record['val_accuracy']:.4f}"
        )

        if val["macro_f1"] > best_f1 + 1e-6:
            best_f1 = val["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            print("Early stopping at epoch", epoch)
            break

    model.load_state_dict(best_state)

    checkpoint_path = checkpoints / f"{run_name}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "tag_names": tag_names,
            "config": {
                "task": task,
                "freeze_embeddings": freeze_embeddings,
                "use_crf": use_crf,
                "bidirectional": bidirectional,
                "dropout": dropout,
                "hidden_size": 128,
                "num_layers": 2,
                "embedding_dim": 100,
            },
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )

    result = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "embedding_coverage": coverage,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "note": (
            "Metrics evaluate agreement with provisional "
            "rule-based annotations, not gold labels."
        ),
    }

    result_path = metrics_dir / f"{run_name}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return model, loaders, vocab, result
