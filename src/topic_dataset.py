"""Urdu news topic dataset for scratch Transformer experiments."""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


TOPICS = [
    "Politics",
    "Sports",
    "Economy",
    "International",
    "Health & Society",
]

TOPIC_TO_ID = {
    topic: i for i, topic in enumerate(TOPICS)
}


def normalize_article_id(value):
    """Normalize integer IDs and explicitly numbered article keys."""
    if isinstance(value, int):
        return value

    text = str(value).strip()

    if text.isdigit():
        return int(text)

    match = re.fullmatch(
        r"(?:article[_\-\s]?)(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return int(match.group(1))

    raise ValueError(
        f"Cannot reliably identify article number from {value!r}"
    )


def align_topic_articles(cleaned_articles, metadata):
    """Match article text to metadata by article number."""
    text_index = {}

    for key, text in cleaned_articles.items():
        article_id = normalize_article_id(key)

        if article_id in text_index:
            raise ValueError(f"Duplicate article ID: {article_id}")

        if not isinstance(text, str):
            raise TypeError(
                f"Article {article_id} must contain text"
            )

        text_index[article_id] = text

    records = []
    seen = set()
    excluded = []

    for item in metadata:
        article_id = normalize_article_id(
            item["article_number"]
        )

        if article_id in seen:
            raise ValueError(
                f"Duplicate metadata article ID: {article_id}"
            )
        seen.add(article_id)

        if article_id not in text_index:
            raise ValueError(
                f"Article {article_id} missing from cleaned corpus"
            )

        topic = item.get("auto_topic")

        if topic not in TOPIC_TO_ID:
            excluded.append({
                "article_id": article_id,
                "reason": "missing_or_invalid_topic",
                "topic": topic,
            })
            continue

        tokens = text_index[article_id].split()

        if not tokens:
            excluded.append({
                "article_id": article_id,
                "reason": "empty_cleaned_text",
            })
            continue

        records.append({
            "article_id": article_id,
            "title": item.get("title", ""),
            "topic": topic,
            "label": TOPIC_TO_ID[topic],
            "needs_review": bool(
                item.get("needs_review", False)
            ),
            "tokens": tokens,
        })

    if set(text_index) != seen:
        raise ValueError(
            "Cleaned corpus and metadata IDs do not match. "
            f"Text-only IDs: {sorted(set(text_index) - seen)[:10]}; "
            f"metadata-only IDs: {sorted(seen - set(text_index))[:10]}"
        )

    return records, excluded


def stratified_article_split(records, seed=42):
    """70/15/15 split, stratified by weak topic label."""
    labels = [r["label"] for r in records]

    train, remainder = train_test_split(
        records,
        test_size=0.30,
        stratify=labels,
        random_state=seed,
    )

    val, test = train_test_split(
        remainder,
        test_size=0.50,
        stratify=[r["label"] for r in remainder],
        random_state=seed,
    )

    splits = {
        "train": train,
        "val": val,
        "test": test,
    }

    id_sets = [
        {r["article_id"] for r in split}
        for split in splits.values()
    ]

    assert not (id_sets[0] & id_sets[1])
    assert not (id_sets[0] & id_sets[2])
    assert not (id_sets[1] & id_sets[2])

    assert sum(map(len, splits.values())) == len(records)

    return splits


def build_topic_vocab(train_records, max_vocab=10000):
    """Build vocabulary using training articles only."""
    counts = Counter(
        token
        for record in train_records
        for token in record["tokens"]
    )

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<CLS>": 2,
    }

    for token, _ in counts.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)

        if len(vocab) >= max_vocab:
            break

    return vocab


class TopicDataset(Dataset):
    def __init__(self, records, vocab, max_length=256):
        self.records = records
        self.vocab = vocab
        self.max_length = max_length

        if max_length < 2:
            raise ValueError("max_length must be >= 2")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        tokens = record["tokens"][:self.max_length - 1]

        ids = [self.vocab["<CLS>"]]
        ids.extend(
            self.vocab.get(token, self.vocab["<UNK>"])
            for token in tokens
        )

        length = len(ids)
        padding = self.max_length - length

        return {
            "input_ids": torch.tensor(
                ids + [self.vocab["<PAD>"]] * padding,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [True] * length + [False] * padding,
                dtype=torch.bool,
            ),
            "label": torch.tensor(
                record["label"],
                dtype=torch.long,
            ),
        }


def save_topic_splits(splits, excluded, output_dir):
    """Save split IDs and labeling caveats for reproducibility."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "topics": TOPICS,
        "split_seed": 42,
        "labels_source": "keyword-derived weak metadata labels",
        "splits": {},
        "excluded": excluded,
    }

    for split, records in splits.items():
        manifest["splits"][split] = [
            {
                "article_id": r["article_id"],
                "topic": r["topic"],
                "needs_review": r["needs_review"],
            }
            for r in records
        ]

    path = output_dir / "topic_splits.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            manifest, f, ensure_ascii=False, indent=2
        )

    return path


def print_topic_distribution(splits):
    for split, records in splits.items():
        counts = Counter(r["topic"] for r in records)
        flagged = sum(r["needs_review"] for r in records)

        print(f"\n{split.upper()} ({len(records)} articles)")
        for topic in TOPICS:
            print(f"  {topic}: {counts[topic]}")

        print("  Needs review:", flagged)
