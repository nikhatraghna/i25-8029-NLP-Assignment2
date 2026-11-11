"""Train BiLSTM NER with/without a scratch linear-chain CRF.

Validation uses exact-span entity F1, not token-level F1.
NER labels may originate from a provisional gazetteer.
"""

import copy
import json
from pathlib import Path

import numpy as np
import torch

from src.bilstm import BiLSTMTagger
from src.ner_evaluation import evaluate_ner_model
from src.sequence_dataset import NER_TAGS
from src.train_bilstm import (
    set_seed,
    load_c3_embeddings,
    build_loaders,
)
from src.sequence_dataset import build_token_vocab


def train_ner(
    splits,
    project_dir,
    use_crf=True,
    freeze_embeddings=False,
    bidirectional=True,
    dropout=0.5,
    random_embeddings=False,
    epochs=40,
    patience=5,
    batch_size=16,
    learning_rate=1e-3,
    weight_decay=1e-4,
    seed=42,
    run_name=None,
):
    set_seed(seed)
    project_dir = Path(project_dir)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    for split in ("train", "val", "test"):
        if split not in splits or not splits[split]:
            raise ValueError(f"Missing NER split: {split}")

    vocab = build_token_vocab(splits["train"])

    if random_embeddings:
        embedding_weights = None
        embedding_info = {"source": "random initialization"}
    else:
        embedding_weights, embedding_info = (
            load_c3_embeddings(project_dir, vocab)
        )

    loaders = build_loaders(
        splits,
        vocab,
        task="ner",
        batch_size=batch_size,
    )

    model = BiLSTMTagger(
        vocab_size=len(vocab),
        num_tags=len(NER_TAGS),
        task="ner",
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
        run_name = (
            f"ner_{'crf' if use_crf else 'no_crf'}_seed{seed}"
        )

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    stale = 0
    history = []

    print("Experiment:", run_name)
    print("Device:", device)
    print("Embedding information:", embedding_info)
    print("NER training sentences:", len(splits["train"]))

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for batch in loaders["train"]:
            x = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)
            tags = batch["tags"].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = model(x, mask, tags)["loss"]

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite NER training loss"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()

            train_losses.append(float(loss.item()))

        val_metrics = evaluate_ner_model(
            model,
            loaders["val"],
            NER_TAGS,
            device,
        )

        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_entity_precision": val_metrics["precision"],
            "val_entity_recall": val_metrics["recall"],
            "val_entity_f1": val_metrics["f1"],
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d} | "
            f"loss={record['train_loss']:.4f} | "
            f"val_entity_f1={record['val_entity_f1']:.4f} | "
            f"P={record['val_entity_precision']:.4f} | "
            f"R={record['val_entity_recall']:.4f}"
        )

        if val_metrics["f1"] > best_f1 + 1e-6:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            stale = 0
        else:
            stale += 1

        if stale >= patience:
            print("Early stopping at epoch", epoch)
            break

    model.load_state_dict(best_state)

    checkpoint_dir = project_dir / "models"
    metrics_dir = project_dir / "results/metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"{run_name}.pt"

    config = {
        "task": "ner",
        "use_crf": use_crf,
        "freeze_embeddings": freeze_embeddings,
        "bidirectional": bidirectional,
        "dropout": dropout,
        "random_embeddings": random_embeddings,
        "embedding_dim": 100,
        "hidden_size": 128,
        "num_layers": 2,
        "seed": seed,
    }

    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "tag_names": NER_TAGS,
            "config": config,
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )

    result = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_val_entity_f1": best_f1,
        "embedding_info": embedding_info,
        "config": config,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "evaluation_note": (
            "Entity-level validation against provisional "
            "gazetteer-generated NER reference labels."
        ),
    }

    with open(
        metrics_dir / f"{run_name}.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result, file,
            ensure_ascii=False,
            indent=2,
        )

    return model, loaders, vocab, result
