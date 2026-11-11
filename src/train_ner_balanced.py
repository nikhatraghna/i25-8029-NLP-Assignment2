"""NER training with entity-aware sentence sampling.

Samples training sentences with/without named entities
approximately equally. Validation remains unmodified.
Uses exact-span entity F1 for checkpoint selection.
"""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.bilstm import BiLSTMTagger
from src.ner_evaluation import evaluate_ner_model
from src.sequence_dataset import (
    NER_TAGS,
    SequenceTaggingDataset,
    make_collate_fn,
    build_token_vocab,
)
from src.train_bilstm import (
    set_seed,
    load_c3_embeddings,
)


def make_balanced_loader(examples, vocab, batch_size=16, seed=42):
    dataset = SequenceTaggingDataset(
        examples, vocab, task="ner"
    )

    has_entity = np.array([
        any(tag != "O" for tag in ex["ner_tags"])
        for ex in examples
    ])

    n_positive = int(has_entity.sum())
    n_negative = len(examples) - n_positive

    if n_positive == 0 or n_negative == 0:
        raise ValueError(
            "Balanced sampling requires both entity-containing "
            "and entity-free training sentences"
        )

    # Approximate 50/50 probability across the two groups.
    weights = np.where(
        has_entity,
        0.5 / n_positive,
        0.5 / n_negative,
    )

    generator = torch.Generator().manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=make_collate_fn("ner"),
    )

    return loader, {
        "train_sentences": len(examples),
        "entity_sentences": n_positive,
        "non_entity_sentences": n_negative,
        "sampling": "approximately 50/50 by sentence group",
    }


def train_balanced_ner(
    splits,
    project_dir,
    use_crf=True,
    epochs=35,
    min_epochs=12,
    patience=6,
    batch_size=16,
    seed=42,
    run_name=None,
):
    set_seed(seed)
    project_dir = Path(project_dir)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if min_epochs > epochs:
        raise ValueError("min_epochs cannot exceed epochs")

    vocab = build_token_vocab(splits["train"])

    vectors, embedding_info = load_c3_embeddings(
        project_dir, vocab, embedding_dim=100
    )

    train_loader, sampling_info = make_balanced_loader(
        splits["train"],
        vocab,
        batch_size=batch_size,
        seed=seed,
    )

    val_dataset = SequenceTaggingDataset(
        splits["val"], vocab, task="ner"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn("ner"),
    )

    model = BiLSTMTagger(
        vocab_size=len(vocab),
        num_tags=len(NER_TAGS),
        task="ner",
        embedding_dim=100,
        embedding_weights=vectors,
        freeze_embeddings=False,
        hidden_size=128,
        num_layers=2,
        dropout=0.5,
        bidirectional=True,
        use_crf=use_crf,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    if run_name is None:
        run_name = (
            "ner_balanced_crf"
            if use_crf else "ner_balanced_no_crf"
        )

    print("Run:", run_name)
    print("Device:", device)
    print("Training distribution:", sampling_info)
    print("C3 embedding coverage:", embedding_info)

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    stale = 0
    history = []

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
                    "Non-finite NER loss"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()

            losses.append(loss.item())

        val = evaluate_ner_model(
            model, val_loader, NER_TAGS, device
        )

        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_entity_precision": val["precision"],
            "val_entity_recall": val["recall"],
            "val_entity_f1": val["f1"],
            "val_tp": val["tp"],
            "val_fp": val["fp"],
            "val_fn": val["fn"],
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d} | "
            f"loss={record['train_loss']:.4f} | "
            f"P={val['precision']:.4f} | "
            f"R={val['recall']:.4f} | "
            f"F1={val['f1']:.4f} | "
            f"TP={val['tp']} FP={val['fp']} FN={val['fn']}"
        )

        if val["f1"] > best_f1 + 1e-6:
            best_f1 = val["f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        # Do not stop before minimum training duration.
        if epoch >= min_epochs and stale >= patience:
            print("Early stopping at epoch", epoch)
            break

    model.load_state_dict(best_state)

    model_dir = project_dir / "models"
    result_dir = project_dir / "results/metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = model_dir / f"{run_name}.pt"

    torch.save({
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "tag_names": NER_TAGS,
        "best_epoch": best_epoch,
        "config": {
            "task": "ner",
            "use_crf": use_crf,
            "embedding_dim": 100,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.5,
            "bidirectional": True,
            "freeze_embeddings": False,
            "balanced_sampling": True,
            "seed": seed,
        },
    }, checkpoint)

    result = {
        "run": run_name,
        "best_epoch": best_epoch,
        "best_val_entity_f1": best_f1,
        "sampling_info": sampling_info,
        "embedding_info": embedding_info,
        "history": history,
        "checkpoint": str(checkpoint),
        "reference_labels": "provisional gazetteer annotations",
    }

    with open(
        result_dir / f"{run_name}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return model, val_loader, vocab, result
