"""Corpus-specific Urdu POS lexicon and contextual rules."""

from collections import Counter
from src.pos_tagger import UrduPOSTagger, create_seed_lexicon


# Reviewed additions derived from the unresolved-token audit.
# These are proposed default tags, not gold annotations.
ADDITIONS = {
    "POST": [
        "نے", "مطابق", "خلاف", "ذریعے",
        "سام", "بارے", "بمقابلہ"
    ],
    "ADV": [
        "بھی", "نہیں", "نہ", "ہی", "صرف",
        "پہلے", "کم", "زیادہ", "دوبارہ",
        "تقریبا", "بالکل", "ابھی", "بعدازاں"
    ],
    "VERB": [
        "ہو", "سکتا", "ہوتا", "گا", "جاتا",
        "کہنا", "کہتا", "دیتا", "جائے",
        "آئا", "بنا", "چکا", "رکھ",
        "کھیل", "حاصل", "استعمال",
        "کرے", "کریں", "ہوئے", "ہوگی",
        "ہوگا", "ہوتی", "ہوتے", "آئے",
        "گئی", "گئے", "کہتے", "کرنا"
    ],
    "PRON": [
        "جو", "جس", "جن", "اسے",
        "ہمارا", "میرا", "انہیں",
        "جسے", "جس نے", "جنہیں"
    ],
    "ADJ": [
        "والا", "پاکستانی", "جیسا",
        "پہلا", "موجود", "اسی",
        "حالیہ", "سابق", "مختلف",
        "مکمل", "عوامی", "سرکاری"
    ],
    "NOUN": [
        "لوگ", "طور", "کپ", "ٹیسٹ",
        "بچہ", "ورلڈ", "پیش",
        "میچ", "سکواڈ", "فیصلہ",
        "کھلاڑی", "اننگز", "رنز",
        "وکٹ", "سکور", "اسکور",
        "وقت", "معاملہ", "تعلق",
        "مسئلہ", "سوال", "حالات",
        "حالت", "خبر", "موقع"
    ],
    "DET": [
        "تمام", "ہر", "کئی", "بعض",
        "کچھ", "کوئی", "چند"
    ]
}


def build_expanded_lexicon():
    lexicon = create_seed_lexicon()

    for tag, words in ADDITIONS.items():
        for word in words:
            # Explicit reviewed additions override old defaults.
            lexicon[word] = tag

    return lexicon


class ContextualUrduPOSTagger(UrduPOSTagger):
    """Seed POS tagger with limited contextual disambiguation."""

    def __init__(self, lemma_table=None):
        super().__init__(
            lexicon=build_expanded_lexicon(),
            lemma_table=lemma_table
        )

    def annotate(self, tokens):
        annotations = []

        for i, token in enumerate(tokens):
            previous = tokens[i - 1] if i > 0 else None
            following = (
                tokens[i + 1]
                if i + 1 < len(tokens) else None
            )

            tag, source = self.predict(token)

            # First-person pronoun versus locative postposition.
            if token == "میں":
                if previous is None or previous in {
                    "کہ", "اگر", "لیکن", "تو", "اور",
                    "جب", "کیونکہ"
                }:
                    tag = "PRON"
                else:
                    tag = "POST"
                source = "context_heuristic"

            # Noun/verb ambiguity in Urdu.
            elif token == "کھیل":
                if following in {
                    "رہا", "رہی", "رہے", "سکا",
                    "سکتی", "سکتا", "کر"
                }:
                    tag = "VERB"
                else:
                    tag = "NOUN"
                source = "context_heuristic"

            # Named modifier versus substantivized adjective.
            elif token == "پاکستانی":
                if following in {
                    "کھلاڑی", "ٹیم", "حکومت",
                    "عوام", "شہری"
                }:
                    tag = "ADJ"
                else:
                    tag = "NOUN"
                source = "context_heuristic"

            annotations.append({
                "token": token,
                "pos": tag,
                "source": source
            })

        return annotations


def annotation_coverage(annotated):
    tag_counts = Counter()
    source_counts = Counter()

    for sentences in annotated.values():
        for sentence in sentences:
            tag_counts.update(sentence["pos_tags"])
            source_counts.update(sentence["pos_sources"])

    total = sum(tag_counts.values())
    unresolved = tag_counts["UNK"]

    return {
        "total_tokens": total,
        "unresolved": unresolved,
        "coverage_percent": round(
            100 * (total - unresolved) / max(total, 1), 2
        ),
        "tag_counts": dict(tag_counts),
        "source_counts": dict(source_counts)
    }
