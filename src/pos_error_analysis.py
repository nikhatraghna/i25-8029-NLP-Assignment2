"""POS test error analysis for the Urdu BiLSTM.

Reports three most frequent gold-to-predicted tag confusions
and up to two real test sentences illustrating each pair.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import torch

from src.sequence_dataset import POS_TAGS


@torch.no_grad()
def collect_pos_errors(model, loader, test_examples, device):
    model.eval()
    errors = []
    offset = 0

    for batch in loader:
        x = batch["input_ids"].to(device)
        mask = batch["mask"].to(device)
        predictions = model.predict(x, mask)

        for i, predicted in enumerate(predictions):
            example = test_examples[offset]
            tokens = example["tokens"]
            gold = example["pos_tags"]

            if not (
                len(tokens) == len(gold) == len(predicted)
            ):
                raise ValueError(
                    f"Token/tag mismatch at test example {offset}"
                )

            for j, (true_tag, pred_id) in enumerate(
                zip(gold, predicted)
            ):
                predicted_tag = POS_TAGS[pred_id]

                if true_tag != predicted_tag:
                    errors.append({
                        "sentence_id": example.get(
                            "sentence_id", str(offset)
                        ),
                        "token_index": j,
                        "token": tokens[j],
                        "gold_tag": true_tag,
                        "predicted_tag": predicted_tag,
                        "sentence": example["sentence"],
                    })

            offset += 1

    if offset != len(test_examples):
        raise ValueError(
            "Test examples and DataLoader order/count differ"
        )

    return errors


def summarize_confusions(errors, top_k=3):
    counts = Counter(
        (e["gold_tag"], e["predicted_tag"])
        for e in errors
    )

    summary = []

    for (gold, pred), count in counts.most_common(top_k):
        matching = [
            e for e in errors
            if e["gold_tag"] == gold
            and e["predicted_tag"] == pred
        ]

        examples = []
        used_sentences = set()

        for error in matching:
            sid = error["sentence_id"]
            if sid in used_sentences:
                continue

            used_sentences.add(sid)
            examples.append(error)

            if len(examples) == 2:
                break

        summary.append({
            "gold_tag": gold,
            "predicted_tag": pred,
            "count": count,
            "examples": examples,
        })

    return summary


def save_analysis(errors, summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "pos_top_confusions.json"
    csv_path = output_dir / "pos_token_errors.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            summary, f, ensure_ascii=False, indent=2
        )

    with csv_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        fields = [
            "sentence_id", "token_index", "token",
            "gold_tag", "predicted_tag", "sentence",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(errors)

    return json_path, csv_path
