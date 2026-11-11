"""Strict alignment of POS and BIO-NER sentence annotations."""

from collections import Counter

from src.sequence_dataset import NER_TAGS, validate_bio


def align_ner_splits(pos_splits, ner_splits):
    """Align NER annotations to POS splits by sentence_id.

    Refuses missing, duplicated or mismatched sentence records.
    Does not silently generate missing labels.
    """
    combined = {}

    for split in ("train", "val", "test"):
        pos_examples = pos_splits[split]
        ner_examples = ner_splits[split]

        ner_index = {}

        for ex in ner_examples:
            sid = ex["sentence_id"]

            if sid in ner_index:
                raise ValueError(
                    f"Duplicate NER sentence_id: {sid}"
                )

            ner_index[sid] = ex

        output = []

        for pos_ex in pos_examples:
            sid = pos_ex["sentence_id"]

            if sid not in ner_index:
                raise ValueError(
                    f"Missing NER annotation for {sid}"
                )

            ner_ex = ner_index[sid]

            if pos_ex["tokens"] != ner_ex["tokens"]:
                raise ValueError(
                    f"Token mismatch for {sid}"
                )

            tags = ner_ex["ner_tags"]

            if len(tags) != len(pos_ex["tokens"]):
                raise ValueError(
                    f"NER length mismatch for {sid}"
                )

            unknown = set(tags) - set(NER_TAGS)
            if unknown:
                raise ValueError(
                    f"Unknown NER labels in {sid}: {unknown}"
                )

            valid, position, label = validate_bio(tags)
            if not valid:
                raise ValueError(
                    f"Invalid BIO in {sid}: "
                    f"position={position}, tag={label}"
                )

            record = dict(pos_ex)
            record["ner_tags"] = list(tags)
            output.append(record)

        if len(output) != len(ner_examples):
            raise ValueError(
                f"Unmatched NER examples in {split}"
            )

        combined[split] = output

    return combined


def ner_label_audit(splits):
    """Summarize reference NER annotations by split."""
    for split, examples in splits.items():
        counts = Counter(
            tag
            for ex in examples
            for tag in ex["ner_tags"]
        )

        sentences_with_entities = sum(
            any(tag != "O" for tag in ex["ner_tags"])
            for ex in examples
        )

        print(f"\n{split.upper()}")
        print("Sentences:", len(examples))
        print("With entities:", sentences_with_entities)
        print("Token label distribution:", dict(counts))
