"""Rule-based Urdu POS annotation for DS5007 Assignment 2."""

import csv
import json
import re
from collections import Counter
from pathlib import Path

POS_TAGS = [
    "NOUN", "VERB", "ADJ", "ADV", "PRON",
    "DET", "CONJ", "POST", "NUM", "PUNC", "UNK"
]

SEED_LEXICON = {
    "PRON": [
        "میں", "ہم", "تم", "آپ", "وہ", "یہ", "اس",
        "ان", "مجھے", "ہمیں", "تمہیں", "انھیں",
        "انھوں", "انہوں", "کون", "کوئی", "خود"
    ],
    "DET": [
        "ایک", "ہر", "کچھ", "کئی", "تمام",
        "سب", "یہی", "وہی", "کسی", "ایسا",
        "اس", "ان", "اپنا", "دوسرا"
    ],
    "CONJ": [
        "اور", "یا", "لیکن", "مگر", "تاہم",
        "بلکہ", "کیونکہ", "اگر", "تو", "کہ",
        "جب", "جبکہ", "چونکہ", "تاکہ"
    ],
    "POST": [
        "کو", "سے", "پر", "تک", "کے", "کی",
        "کا", "میں", "ساتھ", "لیے", "بعد",
        "قبل", "بغیر", "دوران", "تحت", "جانب"
    ],
    "VERB": [
        "ہے", "ہیں", "ہوں", "تھا", "تھی",
        "تھے", "ہوا", "ہوگا", "رہا", "رہی",
        "رہے", "کیا", "کرنا", "کرتا", "کہا",
        "گیا", "آیا", "دیا", "لیا", "ہونا",
        "دیکھ", "کر", "بن", "جا", "آ"
    ],
    "ADV": [
        "آج", "کل", "اب", "پھر", "کبھی",
        "ہمیشہ", "جلد", "فورا", "اکثر",
        "یہاں", "وہاں", "بہت", "تقریبا"
    ],
    "ADJ": [
        "اچھا", "بڑا", "چھوٹا", "نیا",
        "پرانا", "اہم", "سیاسی", "معاشی",
        "قومی", "عالمی", "مقامی", "شدید",
        "مشکل", "آسان", "بہتر", "زیادہ"
    ],
    "NOUN": [
        "پاکستان", "حکومت", "عدالت", "وزیر",
        "معیشت", "فوج", "صحت", "تعلیم",
        "آبادی", "ملک", "شہر", "عوام",
        "کرکٹ", "ٹیم", "کھلاڑی", "قانون",
        "اسکول", "سکول", "ہسپتال", "ڈاکٹر",
        "مریض", "دن", "سال", "وقت", "بات",
        "زندگی", "خبر", "علاقہ", "پانی",
        "تیل", "قیمت", "کاروبار"
    ]
}

# Lexicon conflicts are resolved by explicit priority.
TAG_PRIORITY = [
    "CONJ", "POST", "PRON", "DET",
    "VERB", "ADV", "ADJ", "NOUN"
]

PUNCTUATION = {
    "۔", "،", "؟", "!", "?", ":", ";",
    "؛", "(", ")", "[", "]", "۔"
}

NUM_PATTERN = re.compile(r"^[0-9۰-۹٠-٩]+([.,][0-9۰-۹٠-٩]+)*$")


def create_seed_lexicon():
    """Build a starter lexicon, not the final 200-entry resource."""
    lexicon = {}

    for tag in TAG_PRIORITY:
        for word in SEED_LEXICON.get(tag, []):
            lexicon.setdefault(word, tag)

    return lexicon


class UrduPOSTagger:
    def __init__(self, lexicon=None, lemma_table=None):
        self.lexicon = (
            lexicon if lexicon is not None
            else create_seed_lexicon()
        )
        self.lemma_table = lemma_table or {}

    def predict(self, token):
        """Return a POS tag and its evidence source."""
        if token in PUNCTUATION:
            return "PUNC", "punctuation"

        if token == "<NUM>" or NUM_PATTERN.fullmatch(token):
            return "NUM", "number"

        if token in self.lexicon:
            return self.lexicon[token], "lexicon"

        lemma = self.lemma_table.get(token, token)

        if lemma in self.lexicon:
            return self.lexicon[lemma], "lemma"

        # Heuristic candidates requiring human review.
        if token.endswith(("ترین", "ناک", "مند")):
            return "ADJ", "suffix_heuristic"

        if token.endswith(("تا", "تی", "تے")) and len(token) > 4:
            return "VERB", "suffix_heuristic"

        return "UNK", "unresolved"

    def annotate(self, tokens):
        return [
            {
                "token": token,
                "pos": self.predict(token)[0],
                "source": self.predict(token)[1]
            }
            for token in tokens
        ]


def annotate_splits(splits, tagger):
    """Annotate without modifying the original sentence files."""
    result = {}

    for split_name, sentences in splits.items():
        result[split_name] = []

        for sentence in sentences:
            annotation = tagger.annotate(sentence["tokens"])

            result[split_name].append({
                **sentence,
                "pos_tags": [a["pos"] for a in annotation],
                "pos_sources": [
                    a["source"] for a in annotation
                ]
            })

    return result


def export_annotations(annotated, output_dir):
    """Export provisional CoNLL and review files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, sentences in annotated.items():
        path = output_dir / f"pos_{split_name}.conll"

        with path.open("w", encoding="utf-8") as f:
            for sentence in sentences:
                for token, tag in zip(
                    sentence["tokens"], sentence["pos_tags"]
                ):
                    f.write(f"{token}\t{tag}\n")
                f.write("\n")

    review_path = output_dir / "pos_review.csv"

    with review_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "sentence_id", "topic", "token",
            "predicted_pos", "source", "corrected_pos"
        ])

        for sentences in annotated.values():
            for sentence in sentences:
                for token, tag, source in zip(
                    sentence["tokens"],
                    sentence["pos_tags"],
                    sentence["pos_sources"]
                ):
                    if source in (
                        "suffix_heuristic", "unresolved"
                    ):
                        writer.writerow([
                            sentence["sentence_id"],
                            sentence["topic"],
                            token, tag, source, ""
                        ])


def audit_annotations(annotated, lexicon):
    """Print label coverage and lexicon-size evidence."""
    tags = Counter()
    sources = Counter()

    for sentences in annotated.values():
        for sentence in sentences:
            tags.update(sentence["pos_tags"])
            sources.update(sentence["pos_sources"])

    lexicon_counts = Counter(lexicon.values())

    return {
        "pos_distribution": dict(tags),
        "annotation_sources": dict(sources),
        "lexicon_counts": dict(lexicon_counts),
        "total_tokens": sum(tags.values())
    }
