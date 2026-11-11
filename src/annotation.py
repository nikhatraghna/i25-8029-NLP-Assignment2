"""Sentence sampling for DS5007 POS and NER annotation."""

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split


VALID_TOPICS = {
    "Politics", "Sports", "Economy",
    "International", "Health & Society"
}


def load_topic_metadata(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect_sentences(articles, metadata, min_tokens=5):
    """Collect sentences with article IDs and topic labels."""
    samples = []

    for record in metadata:
        article_id = int(record["article_number"])
        verified = record.get("verified_topic")

        # Accept reviewed labels, or confident weak labels.
        if verified in VALID_TOPICS:
            topic = verified
        elif not record.get("needs_review", True):
            topic = record.get("auto_topic")
        else:
            continue

        if topic not in VALID_TOPICS:
            continue

        if article_id not in articles:
            raise ValueError(f"Missing article {article_id}")

        for sentence_number, line in enumerate(
            articles[article_id].splitlines(), start=1
        ):
            sentence = line.strip()
            tokens = sentence.split()

            if len(tokens) < min_tokens:
                continue

            samples.append({
                "article_id": article_id,
                "sentence_number": sentence_number,
                "topic": topic,
                "sentence": sentence,
                "tokens": tokens
            })

    return samples


def sample_500_sentences(samples, seed=42):
    """Sample 500 sentences, >=100 from each of 3 topics."""
    rng = random.Random(seed)

    by_topic = defaultdict(list)
    for sample in samples:
        by_topic[sample["topic"]].append(sample)

    eligible = [
        topic for topic, rows in by_topic.items()
        if len(rows) >= 100
    ]

    if len(eligible) < 3:
        raise ValueError(
            "Fewer than 3 topics have 100 usable sentences. "
            "Review topic labeling before sampling."
        )

    # Select three topics with the most eligible sentences.
    selected_topics = sorted(
        eligible,
        key=lambda topic: (-len(by_topic[topic]), topic)
    )[:3]

    chosen = []
    for topic in selected_topics:
        chosen.extend(rng.sample(by_topic[topic], 100))

    chosen_keys = {
        (r["article_id"], r["sentence_number"])
        for r in chosen
    }

    remaining = [
        r for topic in selected_topics
        for r in by_topic[topic]
        if (r["article_id"], r["sentence_number"])
        not in chosen_keys
    ]

    if len(remaining) < 200:
        raise ValueError("Not enough remaining sentences.")

    chosen.extend(rng.sample(remaining, 200))
    rng.shuffle(chosen)

    for idx, item in enumerate(chosen, start=1):
        item["sentence_id"] = idx

    assert len(chosen) == 500
    counts = Counter(r["topic"] for r in chosen)
    assert all(counts[t] >= 100 for t in selected_topics)

    return chosen, selected_topics


def split_sentences(samples, seed=42):
    """Create 70/15/15 stratified sentence-level splits."""
    topics = [s["topic"] for s in samples]

    train, temp = train_test_split(
        samples,
        test_size=0.30,
        random_state=seed,
        stratify=topics
    )

    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=seed,
        stratify=[s["topic"] for s in temp]
    )

    assert (len(train), len(val), len(test)) == (
        350, 75, 75
    )

    return {"train": train, "val": val, "test": test}


def save_sampling_results(splits, output_dir):
    """Save sentence records for subsequent POS/NER labeling."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, rows in splits.items():
        path = output_dir / f"sentences_{split_name}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                rows, f, ensure_ascii=False, indent=2
            )

    with open(
        output_dir / "sentence_distribution.csv",
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["split", "topic", "count"])

        for split_name, rows in splits.items():
            for topic, count in sorted(
                Counter(r["topic"] for r in rows).items()
            ):
                writer.writerow([split_name, topic, count])
