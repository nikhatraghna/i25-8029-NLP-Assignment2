"""Controlled NER ablations with entity-aware training sampling.

A1: Unidirectional LSTM
A2: No dropout
A3: Random embeddings
A4: No CRF

All experiments use the same data, seed, optimizer,
training duration, and validation entity-level metric.
"""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.bilstm import BiLSTMTagger
from src.ner_evaluation import evaluate_ner_model
from src.sequence_dataset import (
    NER_TAGS,
    SequenceTaggingDataset,
    make_collate_fn,
    build_token_vocab,
)
from src.train_bilstm import set_seed, load_c3_embeddings
from src.train_ner_balanced import make_balanced_loader


ABLATIONS = {
    "A1_unidirectional": {
        "bidirectional": False,
        "dropout": 0.5,
        "random_embeddings": False,
        "use_crf": True,
    },
    "A2_no_dropout": {
        "bidirectional": True,
        "dropout": 0.0,
        "random_embeddings": False,
        "use_crf": True,
    },
    "A3_random_embeddings": {
        "bidirectional": True,
        "dropout": 0.5,
        "random_embeddings": True,
        "use_crf": True,
    },
    "A4_no_crf": {
        "bidirectional": True,
        "dropout": 0.5,
        "random_embeddings": False,
        "use_crf": False,
    },
}


def run_ablation(
    splits,
    project_dir,
    name,
    seed=42,
    epochs=35,
    min_epochs=12,
    patience=6,
    batch_size=16,
):
    if name not in ABLATIONS:
        raise ValueError(f"Unknown experiment: {name}")

    set_seed(seed)
    config = ABLATIONS[name]
    project_dir = Path(project_dir)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    vocab = build_token_vocab(splits["train"])

    if config["random_embeddings"]:
        weights = None
        embedding_info = {"source": "random"}
    else:
        weights, embedding_info = load_c3_embeddings(
            project_dir, vocab, embedding_dim=100
        )

    train_loader, sampling_info = make_balanced_loader(
        splits["train"],
        vocab,
        batch_size=batch_size,
        seed=seed,
    )

    val_loader = DataLoader(
        SequenceTaggingDataset(
            splits["val"],
            vocab,
            task="ner",
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn("ner"),
    )

    model = BiLSTMTagger(
        vocab_size=len(vocab),
        num_tags=len(NER_TAGS),
        task="ner",
        embedding_dim=100,
        hidden_size=128,
        num_layers=2,
        dropout=config["dropout"],
        bidirectional=config["bidirectional"],
        embedding_weights=weights,
        freeze_embeddings=False,
        use_crf=config["use_crf"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    history = []

    print(f"\nExperiment: {name}")
    print("Device:", device)
    print("Configuration:", config)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for batch in train_loader:
            x = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)
            tags = batch["tags"].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = model(x, mask, tags)["loss"]

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss in {name}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()

            losses.append(float(loss.item()))

        val = evaluate_ner_model(
            model, val_loader, NER_TAGS, device
        )

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_precision": val["precision"],
            "val_recall": val["recall"],
            "val_entity_f1": val["f1"],
            "val_tp": val["tp"],
            "val_fp": val["fp"],
            "val_fn": val["fn"],
        }
        history.append(row)

        print(
            f"{name} | epoch {epoch:02d} | "
            f"loss={row['train_loss']:.4f} | "
            f"entity_F1={row['val_entity_f1']:.4f}"
        )

        if val["f1"] > best_f1 + 1e-6:
            best_f1 = val["f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            stale = 0
        else:
            stale += 1

        if epoch >= min_epochs and stale >= patience:
            print("Early stopping at epoch", epoch)
            break

    model.load_state_dict(best_state)

    model_dir = project_dir / "models"
    metrics_dir = project_dir / "results/metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = model_dir / f"ner_{name}.pt"

    torch.save({
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "tag_names": NER_TAGS,
        "best_epoch": best_epoch,
        "config": {
            **config,
            "embedding_dim": 100,
            "hidden_size": 128,
            "num_layers": 2,
            "seed": seed,
            "balanced_sampling": True,
        },
    }, checkpoint_path)

    result = {
        "experiment": name,
        "best_epoch": best_epoch,
        "best_val_entity_f1": best_f1,
        "configuration": config,
        "sampling_info": sampling_info,
        "embedding_info": embedding_info,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "label_provenance": "provisional gazetteer annotations",
    }

    with (
        metrics_dir / f"ner_{name}.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            result, f, ensure_ascii=False, indent=2
        )

    return result
