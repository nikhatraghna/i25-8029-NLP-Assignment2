"""Held-out evaluation for the scratch Urdu news Transformer."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

from src.topic_dataset import TOPICS


@torch.no_grad()
def evaluate_test(model, loader, device):
    model.eval()
    gold, predicted = [], []

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        logits = model(ids, mask)
        predictions = logits.argmax(dim=-1)

        gold.extend(batch["label"].tolist())
        predicted.extend(predictions.cpu().tolist())

    if not gold:
        raise ValueError("Test loader is empty")

    labels = list(range(len(TOPICS)))

    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(
            f1_score(
                gold, predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            gold, predicted, labels=labels
        ).tolist(),
        "classification_report": classification_report(
            gold, predicted,
            labels=labels,
            target_names=TOPICS,
            output_dict=True,
            zero_division=0,
        ),
        "y_true": gold,
        "y_pred": predicted,
    }


def save_test_results(results, project_dir):
    project_dir = Path(project_dir)
    metrics_dir = project_dir / "results/metrics"
    plots_dir = project_dir / "results/plots"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = (
        metrics_dir / "transformer_test_metrics.json"
    )

    output = {
        **results,
        "label_provenance": (
            "Keyword-derived topic labels; some articles "
            "were flagged for manual review."
        ),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            output, f, ensure_ascii=False, indent=2
        )

    fig, ax = plt.subplots(figsize=(9, 7))

    cm = np.array(results["confusion_matrix"])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=TOPICS,
    )
    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=35,
        colorbar=False,
        values_format="d",
    )

    ax.set_title("Scratch Transformer — Test Confusion Matrix")
    fig.tight_layout()

    plot_path = (
        plots_dir / "transformer_test_confusion.png"
    )
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return metrics_path, plot_path
