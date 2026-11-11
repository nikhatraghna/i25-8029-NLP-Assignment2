"""Corpus-specific Urdu POS lexicon and contextual tagging rules.

DS5007 Assignment 2
Uses the POS tagger implemented in src/pos_tagger.py.

The lexicon contains reviewed seed entries and proposed
corpus-specific additions. Context-sensitive rules are
heuristics and must be audited before final training.
"""

from collections import Counter

from src.pos_tagger import (
    UrduPOSTagger,
    create_seed_lexicon
)


# ============================================================
# 1. CORPUS-SPECIFIC POS VOCABULARY
# ============================================================

ADDITIONS = {
    "POST": [
        "نے", "مطابق", "خلاف", "ذریعے",
        "سام", "بارے", "بمقابلہ"
    ],

    "ADV": [
        "بھی", "نہیں", "نہ", "ہی", "صرف",
        "پہلے", "کم", "زیادہ", "دوبارہ",
        "تقریبا", "بالکل", "ابھی",
        "بعدازاں"
    ],

    "VERB": [
        "ہو", "سکتا", "ہوتا", "گا",
        "جاتا", "کہنا", "کہتا", "دیتا",
        "جائے", "آئا", "بنا", "چکا",
        "رکھ", "کھیل", "حاصل",
        "استعمال", "کرے", "کریں",
        "ہوئے", "ہوگی", "ہوگا",
        "ہوتی", "ہوتے", "آئے",
        "گئی", "گئے", "کہتے",
        "کرنا"
    ],

    "PRON": [
        "جو", "جس", "جن", "اسے",
        "ہمارا", "میرا", "انہیں",
        "جسے", "جنہیں"
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
        "تمام", "ہر", "کئی",
        "بعض", "کچھ", "کوئی", "چند"
    ]
}


# ============================================================
# 2. NEW CORRECTIONS FROM ACTUAL ANNOTATION EXAMPLES
# ============================================================

EXAMPLE_CORRECTIONS = {
    "NOUN": [
        "طرح",
        "سٹیجز",
        "انگلینڈ",
        "ویلز",
        "بورڈ",
        "واقعہ",
        "ادارہ",
        "ریگولیٹر",
        "حوالہ",
        "تحقیق"
    ],

    "VERB": [
        "پرفارم",
        "کرا"
    ],

    "ADJ": [
        "خاص",
        "آزاد",
        "نگران"
    ],

    "CONJ": [
        "اینڈ"
    ]
}


# ============================================================
# 3. BUILD EXPANDED LEXICON
# ============================================================

def build_expanded_lexicon():
    """Combine the original seed lexicon with new entries."""

    lexicon = create_seed_lexicon()

    # First apply corpus-specific vocabulary.
    for tag, words in ADDITIONS.items():
        for word in words:
            lexicon[word] = tag

    # Apply reviewed corrections last.
    for tag, words in EXAMPLE_CORRECTIONS.items():
        for word in words:
            lexicon[word] = tag

    return lexicon


# ============================================================
# 4. CONTEXT-AWARE POS TAGGER
# ============================================================

class ContextualUrduPOSTagger(UrduPOSTagger):
    """Rule-based POS tagger with limited contextual rules."""

    def __init__(self, lemma_table=None):
        super().__init__(
            lexicon=build_expanded_lexicon(),
            lemma_table=lemma_table
        )

    def annotate(self, tokens):
        annotations = []

        for i, token in enumerate(tokens):

            previous = (
                tokens[i - 1] if i > 0 else None
            )

            following = (
                tokens[i + 1]
                if i + 1 < len(tokens) else None
            )

            tag, source = self.predict(token)

            # ------------------------------------------------
            # Rule 1: "میں" — pronoun versus postposition
            # ------------------------------------------------
            if token == "میں":

                if previous is None or previous in {
                    "کہ", "اگر", "لیکن",
                    "تو", "اور", "جب",
                    "کیونکہ"
                }:
                    tag = "PRON"
                else:
                    tag = "POST"

                source = "context_heuristic"

            # ------------------------------------------------
            # Rule 2: "کھیل" — noun versus verb
            # ------------------------------------------------
            elif token == "کھیل":

                if following in {
                    "رہا", "رہی", "رہے",
                    "سکا", "سکتی", "سکتا",
                    "کر"
                }:
                    tag = "VERB"
                else:
                    tag = "NOUN"

                source = "context_heuristic"

            # ------------------------------------------------
            # Rule 3: "پاکستانی" — adjective versus noun
            # ------------------------------------------------
            elif token == "پاکستانی":

                if following in {
                    "کھلاڑی", "ٹیم",
                    "حکومت", "عوام",
                    "شہری", "فوج",
                    "کرکٹ"
                }:
                    tag = "ADJ"
                else:
                    tag = "NOUN"

                source = "context_heuristic"

            # ------------------------------------------------
            # Rule 4: "والا" as an adjectival modifier
            # ------------------------------------------------
            elif token == "والا":
                tag = "ADJ"
                source = "context_heuristic"

            # ------------------------------------------------
            # Rule 5: English conjunction transliterated in Urdu
            # ------------------------------------------------
            elif token == "اینڈ":
                tag = "CONJ"
                source = "context_heuristic"

            annotations.append({
                "token": token,
                "pos": tag,
                "source": source
            })

        return annotations


# ============================================================
# 5. ANNOTATION COVERAGE AUDIT
# ============================================================

def annotation_coverage(annotated):
    """Calculate the distribution of generated POS labels."""

    tag_counts = Counter()
    source_counts = Counter()

    for sentences in annotated.values():
        for sentence in sentences:

            tag_counts.update(sentence["pos_tags"])
            source_counts.update(sentence["pos_sources"])

    total = sum(tag_counts.values())
    unresolved = tag_counts["UNK"]

    coverage = (
        100 * (total - unresolved) / total
        if total else 0.0
    )

    return {
        "total_tokens": total,
        "unresolved": unresolved,
        "coverage_percent": round(coverage, 2),
        "tag_counts": dict(tag_counts),
        "source_counts": dict(source_counts)
    }


# ============================================================
# 6. LEXICON SIZE AUDIT
# ============================================================

def lexicon_statistics():
    """Report handcrafted vocabulary entries per POS category."""

    lexicon = build_expanded_lexicon()

    counts = Counter(lexicon.values())

    return {
        "total_entries": len(lexicon),
        "entries_per_tag": dict(counts)
    }
