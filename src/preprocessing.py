import json
import re
from pathlib import Path
from collections import Counter

ARTICLE_PATTERN = re.compile(
    r"^### Article (\d+) ###\s*$",
    re.MULTILINE
)


def load_articles(file_path):
    text = Path(file_path).read_text(encoding="utf-8")
    parts = ARTICLE_PATTERN.split(text)

    return {
        int(parts[i]): parts[i + 1].strip()
        for i in range(1, len(parts) - 1, 2)
    }


def load_metadata(file_path):
    with open(file_path, encoding="utf-8") as f:
        records = json.load(f)

    return {
        int(record["article_number"]): record
        for record in records
    }


def tokenize(text):
    return text.split()


def build_vocabulary(articles, max_vocab=10000):
    frequencies = Counter()

    for article in articles.values():
        frequencies.update(tokenize(article))

    special_tokens = ["<PAD>", "<UNK>"]

    most_common = [
        word for word, _ in frequencies.most_common()
        if word not in special_tokens
    ][:max_vocab - len(special_tokens)]

    vocabulary = special_tokens + most_common

    return {
        word: idx for idx, word in enumerate(vocabulary)
    }, frequencies


def verify_alignment(cleaned, raw, metadata):
    ids = set(cleaned)

    if ids != set(raw) or ids != set(metadata):
        raise ValueError("Article IDs do not match.")

    return True
