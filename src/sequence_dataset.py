"""Dataset utilities for Urdu POS and BIO-NER sequence tagging."""

from collections import Counter

import torch
from torch.utils.data import Dataset


POS_TAGS = [
    "NOUN", "VERB", "ADJ", "ADV", "PRON",
    "DET", "CONJ", "POST", "NUM", "PUNC", "UNK"
]

NER_TAGS = [
    "O",
    "B-PER", "I-PER",
    "B-LOC", "I-LOC",
    "B-ORG", "I-ORG",
    "B-MISC", "I-MISC",
]


def build_token_vocab(training_examples, min_freq=1):
    """Build a vocabulary from training tokens only."""
    counts = Counter(
        token
        for ex in training_examples
        for token in ex["tokens"]
    )

    vocab = {"<PAD>": 0, "<UNK>": 1}

    for token, frequency in sorted(
        counts.items(), key=lambda x: (-x[1], x[0])
    ):
        if frequency >= min_freq and token not in vocab:
            vocab[token] = len(vocab)

    return vocab


def validate_examples(examples, tag_key, allowed_tags):
    """Check token/tag alignment and label validity."""
    allowed = set(allowed_tags)

    if not examples:
        raise ValueError("No examples supplied")

    for i, ex in enumerate(examples):
        if "tokens" not in ex or tag_key not in ex:
            raise KeyError(
                f"Example {i}: requires tokens and {tag_key}"
            )

        tokens = ex["tokens"]
        tags = ex[tag_key]

        if not tokens:
            raise ValueError(f"Example {i} is empty")

        if len(tokens) != len(tags):
            raise ValueError(
                f"Example {i}: {len(tokens)} tokens "
                f"but {len(tags)} tags"
            )

        invalid = set(tags) - allowed
        if invalid:
            raise ValueError(
                f"Example {i}: invalid labels {invalid}"
            )


def validate_bio(tags):
    """Check that I- tags follow a compatible B- or I- tag."""
    previous = "O"

    for i, tag in enumerate(tags):
        if tag.startswith("I-"):
            entity_type = tag[2:]
            valid_predecessors = {
                "B-" + entity_type,
                "I-" + entity_type,
            }
            if previous not in valid_predecessors:
                return False, i, tag

        previous = tag

    return True, None, None


class SequenceTaggingDataset(Dataset):
    def __init__(
        self,
        examples,
        vocab,
        task="pos",
        validate_bio_sequences=True,
    ):
        if task not in {"pos", "ner"}:
            raise ValueError("task must be pos or ner")

        self.examples = examples
        self.vocab = vocab
        self.task = task

        self.tag_key = (
            "pos_tags" if task == "pos" else "ner_tags"
        )
        self.tag_names = (
            POS_TAGS if task == "pos" else NER_TAGS
        )
        self.tag_to_id = {
            tag: i for i, tag in enumerate(self.tag_names)
        }

        validate_examples(
            examples,
            self.tag_key,
            self.tag_names,
        )

        if task == "ner" and validate_bio_sequences:
            for i, ex in enumerate(examples):
                valid, position, tag = validate_bio(
                    ex["ner_tags"]
                )
                if not valid:
                    raise ValueError(
                        f"Invalid BIO sequence at example {i}, "
                        f"position {position}: {tag}"
                    )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        ex = self.examples[index]

        return {
            "input_ids": [
                self.vocab.get(token, 1)
                for token in ex["tokens"]
            ],
            "tags": [
                self.tag_to_id[tag]
                for tag in ex[self.tag_key]
            ],
            "tokens": ex["tokens"],
        }


def make_collate_fn(task="pos", pad_id=0):
    """Pad batches and create masks."""
    if task not in {"pos", "ner"}:
        raise ValueError("Unknown task")

    tag_pad = -100 if task == "pos" else 0

    def collate(batch):
        batch_size = len(batch)
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids = torch.full(
            (batch_size, max_len),
            pad_id,
            dtype=torch.long,
        )
        tags = torch.full(
            (batch_size, max_len),
            tag_pad,
            dtype=torch.long,
        )
        mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.bool,
        )

        for i, item in enumerate(batch):
            length = len(item["input_ids"])

            input_ids[i, :length] = torch.tensor(
                item["input_ids"], dtype=torch.long
            )
            tags[i, :length] = torch.tensor(
                item["tags"], dtype=torch.long
            )
            mask[i, :length] = True

        return {
            "input_ids": input_ids,
            "tags": tags,
            "mask": mask,
        }

    return collate
