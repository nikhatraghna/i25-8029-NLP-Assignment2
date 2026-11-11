"""Train an independent scratch BiLSTM news classifier."""

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.bilstm_classifier import UrduNewsBiLSTM
from src.topic_dataset import TOPICS
from src.train_transformer import (
    set_seed,
    make_topic_loaders,
    make_warmup_cosine_scheduler,
    evaluate_classifier,
)


def train_bilstm_classifier(
    splits,
    vocab,
    project_dir,
    epochs=20,
    batch_size=8,
    learning_rate=5e-4,
    weight_decay=0.01,
    warmup_steps=50,
    seed=42,
    run_name="bilstm_urdu_5topics",
):
    set_seed(seed)

    project_dir = Path(project_dir)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    loaders = make_topic_loaders(
        splits,
        vocab,
        batch_size=batch_size,
        max_length=256,
    )

    model = UrduNewsBiLSTM(
        vocab_size=len(vocab),
        num_classes=len(TOPICS),
        embedding_dim=128,
        hidden_size=128,
        num_layers=2,
        dropout=0.5,
        pad_id=vocab["<PAD>"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_steps = epochs * len(loaders["train"])

    scheduler = make_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    history = []
    best_f1 = -1.0
    best_state = None
    best_epoch = None

    print("Experiment:", run_name)
    print("Device:", device)
    print("Training examples:", len(splits["train"]))
    print("Validation examples:", len(splits["val"]))
    print("Total training steps:", total_steps)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for batch in loaders["train"]:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(ids, mask)
            loss = F.cross_entropy(logits, labels)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite classification loss"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())

        val = evaluate_classifier(
            model,
            loaders["val"],
            device,
        )

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss={row['train_loss']:.4f} | "
            f"val_acc={row['val_accuracy']:.4f} | "
            f"val_macro_f1={row['val_macro_f1']:.4f}"
        )

        if val["macro_f1"] > best_f1 + 1e-8:
            best_f1 = val["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )

    model.load_state_dict(best_state)

    model_dir = project_dir / "models"
    metrics_dir = project_dir / "results/metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = model_dir / f"{run_name}.pt"

    config = {
        "model": "scratch_bilstm",
        "embedding_dim": 128,
        "hidden_size": 128,
        "num_layers": 2,
        "bidirectional": True,
        "dropout": 0.5,
        "num_classes": len(TOPICS),
        "max_length": 256,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "seed": seed,
    }

    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "topics": TOPICS,
            "config": config,
            "best_epoch": best_epoch,
        },
        checkpoint,
    )

    result = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "configuration": config,
        "history": history,
        "checkpoint": str(checkpoint),
        "label_provenance": "Keyword-derived weak topic labels",
    }

    metrics_path = metrics_dir / f"{run_name}.json"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return model, loaders, result
