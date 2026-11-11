"""Review and validate a handcrafted Urdu POS lexicon."""

import csv
from collections import Counter
from pathlib import Path

from src.pos_tagger import POS_TAGS


def generate_review_file(annotations, output_path):
    """Create a corpus-specific lexicon review sheet."""
    unresolved = Counter()
    examples = {}

    for sentences in annotations.values():
        for sentence in sentences:
            for token, tag in zip(
                sentence["tokens"], sentence["pos_tags"]
            ):
                if tag == "UNK":
                    unresolved[token] += 1
                    examples.setdefault(
                        token, sentence["sentence"]
                    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    # Preserve previously entered reviews.
    previous = {}
    if output_path.exists():
        with output_path.open(
            encoding="utf-8-sig", newline=""
        ) as f:
            for row in csv.DictReader(f):
                previous[row["token"]] = row.get(
                    "verified_pos", ""
                )

    with output_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "token", "frequency", "example_sentence",
            "verified_pos"
        ])

        for token, frequency in unresolved.most_common():
            writer.writerow([
                token,
                frequency,
                examples[token],
                previous.get(token, "")
            ])

    return len(unresolved)


def load_reviewed_lexicon(review_path):
    """Only accept explicitly reviewed valid POS entries."""
    reviewed = {}

    with open(
        review_path, encoding="utf-8-sig", newline=""
    ) as f:
        for row in csv.DictReader(f):
            token = row["token"].strip()
            tag = row["verified_pos"].strip().upper()

            if not tag:
                continue

            if tag not in POS_TAGS or tag == "UNK":
                raise ValueError(
                    f"Invalid POS tag for {token}: {tag}"
                )

            reviewed[token] = tag

    return reviewed


def lexicon_audit(lexicon, major_tags=None):
    """Check per-tag counts against the assignment requirement."""
    if major_tags is None:
        major_tags = ["NOUN", "VERB", "ADJ", "ADV"]

    counts = Counter(lexicon.values())

    print("Total unique lexicon entries:", len(lexicon))
    print("\nEntries per POS category:")

    for tag in POS_TAGS:
        print(f"{tag:5s}: {counts[tag]}")

    print("\nMajor category requirement:")
    complete = True

    for tag in major_tags:
        count = counts[tag]
        passed = count >= 200
        complete = complete and passed

        print(
            f"{tag}: {count}/200 | "
            f"{'PASS' if passed else 'INCOMPLETE'}"
        )

    return complete, counts
