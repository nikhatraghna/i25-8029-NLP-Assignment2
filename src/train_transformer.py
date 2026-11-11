"""Scratch Transformer training for five-class Urdu news classification.

- AdamW optimizer
- 50-step linear warmup followed by cosine decay
- 20 training epochs
- Best validation macro-F1 checkpoint
- No test-set evaluation during training
"""

import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from src.transformer_classifier import UrduNewsTransformer
from src.topic_dataset import TopicDataset, TOPICS


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_topic_loaders(
    splits,
    vocab,
    batch_size=8,
    max_length=256,
):
    loaders = {}

    for name in ("train", "val", "test"):
        dataset = TopicDataset(
            splits[name],
            vocab,
            max_length=max_length,
        )

        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=0,
        )

    return loaders


def make_warmup_cosine_scheduler(
    optimizer,
    total_steps,
    warmup_steps=50,
):
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    if warmup_steps >= total_steps:
        raise ValueError(
            "Warmup steps must be smaller than total steps"
        )

    def lr_factor(step):
        # LambdaLR performs its initial step during setup.
        # Adding 1 gives the first optimizer update a nonzero LR.
        if step < warmup_steps:
            return (step + 1) / warmup_steps

        progress = (
            (step - warmup_steps)
            / max(1, total_steps - warmup_steps)
        )

        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_factor)


@torch.no_grad()
def evaluate_classifier(model, loader, device):
    model.eval()

    gold = []
    predicted = []
    losses = []

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(ids, mask)
        loss = F.cross_entropy(logits, labels)

        losses.append(float(loss.item()))
        gold.extend(labels.cpu().tolist())
        predicted.extend(
            logits.argmax(dim=-1).cpu().tolist()
        )

    if not gold:
        raise ValueError("Evaluation dataset is empty")

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(
            accuracy_score(gold, predicted)
        ),
        "macro_f1": float(
            f1_score(
                gold,
                predicted,
                labels=list(range(len(TOPICS))),
                average="macro",
                zero_division=0,
            )
        ),
        "y_true": gold,
        "y_pred": predicted,
    }


def train_transformer(
    splits,
    vocab,
    project_dir,
    epochs=20,
    batch_size=8,
    learning_rate=5e-4,
    weight_decay=0.01,
    warmup_steps=50,
    seed=42,
    run_name="transformer_urdu_5topics",
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

    model = UrduNewsTransformer(
        vocab_size=len(vocab),
        num_classes=5,
        d_model=128,
        num_heads=4,
        num_layers=4,
        ff_dim=512,
        max_length=256,
        dropout=0.1,
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

    print("Experiment:", run_name)
    print("Device:", device)
    print("Vocabulary:", len(vocab))
    print("Training articles:", len(splits["train"]))
    print("Validation articles:", len(splits["val"]))
    print("Test articles:", len(splits["test"]))
    print("Total optimizer steps:", total_steps)
    print("Warmup steps:", warmup_steps)

    best_f1 = -1.0
    best_epoch = 0
    best_state = None

    global_step = 0
    history = []

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
                    "Non-finite Transformer training loss"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()
            global_step += 1

            train_losses.append(float(loss.item()))

        validation = evaluate_classifier(
            model,
            loaders["val"],
            device,
        )

        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
            "val_macro_f1": validation["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        history.append(record)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={record['train_loss']:.4f} | "
            f"val_loss={record['val_loss']:.4f} | "
            f"val_acc={record['val_accuracy']:.4f} | "
            f"val_macro_f1={record['val_macro_f1']:.4f}"
        )

        if validation["macro_f1"] > best_f1 + 1e-8:
            best_f1 = validation["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    # Restore the best validation checkpoint.
    model.load_state_dict(best_state)

    model_dir = project_dir / "models"
    metrics_dir = project_dir / "results/metrics"

    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = model_dir / f"{run_name}.pt"

    config = {
        "num_classes": 5,
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 4,
        "ff_dim": 512,
        "max_length": 256,
        "dropout": 0.1,
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
        checkpoint_path,
    )

    result = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "configuration": config,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "label_provenance": (
            "Five keyword-derived article topics; "
            "not independently verified ground truth"
        ),
    }

    metrics_path = metrics_dir / f"{run_name}.json"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return model, loaders, result
