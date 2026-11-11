"""Attention visualization for the scratch Urdu Transformer.

Select three correctly classified held-out test articles.
Visualize two final-layer attention heads per article.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.topic_dataset import TOPICS, TopicDataset


@torch.no_grad()
def collect_correct_articles(
    model,
    test_records,
    vocab,
    device,
    num_examples=3,
):
    """Find correctly classified articles without changing the model."""
    model.eval()

    dataset = TopicDataset(
        test_records,
        vocab,
        max_length=256,
    )

    selected = []

    for index, record in enumerate(test_records):
        example = dataset[index]

        ids = example["input_ids"].unsqueeze(0).to(device)
        mask = example["attention_mask"].unsqueeze(0).to(device)

        logits, attention = model(
            ids,
            mask,
            return_attention=True,
        )

        prediction = int(logits.argmax(dim=-1).item())
        gold = int(example["label"].item())

        if prediction != gold:
            continue

        valid_length = int(mask.sum().item())

        # First 256 tokens include CLS.
        token_labels = ["<CLS>"] + record["tokens"][:255]

        if len(token_labels) != valid_length:
            raise ValueError(
                "Mismatch between token labels and attention mask"
            )

        selected.append({
            "article_id": record["article_id"],
            "title": record.get("title", ""),
            "topic": TOPICS[gold],
            "predicted_topic": TOPICS[prediction],
            "needs_review": record.get("needs_review", False),
            "tokens": token_labels,
            "attention": attention[0].cpu().numpy(),
        })

        if len(selected) == num_examples:
            break

    return selected


def save_attention_heatmaps(
    selected,
    output_dir,
    heads=(0, 1),
    max_visual_tokens=24,
):
    """Save one heatmap for each article/head pair.

    Displays CLS plus the first max_visual_tokens-1 article
    tokens. Attention values are from the full 256-token input,
    so the displayed matrix is a cropped view.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    output_paths = []

    for article in selected:
        tokens = article["tokens"]
        attention = article["attention"]

        visible_length = min(
            len(tokens),
            max_visual_tokens,
        )

        for head in heads:
            if not 0 <= head < attention.shape[0]:
                raise ValueError(f"Invalid head index: {head}")

            matrix = attention[
                head,
                :visible_length,
                :visible_length,
            ]

            fig, ax = plt.subplots(figsize=(11, 9))

            image = ax.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
                vmin=0,
                vmax=max(0.05, float(matrix.max())),
            )

            tick_labels = [
                f"{i}: {token}"
                for i, token in enumerate(tokens[:visible_length])
            ]

            ax.set_xticks(range(visible_length))
            ax.set_yticks(range(visible_length))
            ax.set_xticklabels(
                tick_labels,
                rotation=90,
                fontsize=8,
            )
            ax.set_yticklabels(
                tick_labels,
                fontsize=8,
            )

            ax.set_xlabel("Key token")
            ax.set_ylabel("Query token")
            ax.set_title(
                f"Article {article['article_id']} | "
                f"{article['topic']} | "
                f"Final layer, head {head + 1}"
            )

            fig.colorbar(
                image,
                ax=ax,
                label="Attention probability",
                fraction=0.045,
            )

            fig.tight_layout()

            filename = (
                f"attention_article_{article['article_id']}"
                f"_head_{head + 1}.png"
            )
            destination = output_dir / filename

            fig.savefig(
                destination,
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

            output_paths.append(str(destination))

            records.append({
                "article_id": article["article_id"],
                "title": article["title"],
                "topic": article["topic"],
                "predicted_topic": article["predicted_topic"],
                "needs_review": article["needs_review"],
                "head": head + 1,
                "visible_tokens": visible_length,
                "path": str(destination),
                "note": (
                    "Cropped view of final-layer attention "
                    "computed over the full article input."
                ),
            })

    return output_paths, records


def save_attention_manifest(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            records,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return path
