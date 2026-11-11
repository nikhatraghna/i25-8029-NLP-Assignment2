"""Gazetteer-based Urdu BIO NER tagging for DS5007."""

import csv
import json
from collections import Counter
from pathlib import Path


PERSONS = """
عمران خان
نواز شریف
شہباز شریف
مریم نواز
بلاول بھٹو
آصف علی زرداری
بینظیر بھٹو
ذوالفقار علی بھٹو
محمد علی جناح
فاطمہ جناح
علامہ اقبال
لیاقت علی خان
خواجہ ناظم الدین
اسکندر مرزا
ایوب خان
یحیی خان
ذوالفقار بھٹو
ضیا الحق
پرویز مشرف
عمران اسماعیل
شاہد خاقان عباسی
یوسف رضا گیلانی
راجہ پرویز اشرف
چوہدری نثار
اسحاق ڈار
احسن اقبال
خواجہ آصف
محسن نقوی
علی امین گنڈاپور
عطا تارڑ
حنا ربانی کھر
شیری رحمان
ملالہ یوسفزئی
عبدالستار ایدھی
بلقیس ایدھی
عبدالقدیر خان
عبدالسلام
ارفع کریم
جہانگیر خان
جان شیر خان
وسیم اکرم
وقار یونس
شاہد آفریدی
بابر اعظم
محمد رضوان
شعیب اختر
سرفراز احمد
نسیم شاہ
شاہین آفریدی
حسن علی
محمد حفیظ
مصباح الحق
یونس خان
انضمام الحق
جاوید میانداد
عاطف اسلم
راحت فتح علی خان
نصرت فتح علی خان
علی ظفر
ماہرہ خان
""".strip().splitlines()


LOCATIONS = """
پاکستان
اسلام آباد
لاہور
کراچی
پشاور
کوئٹہ
راولپنڈی
فیصل آباد
ملتان
حیدرآباد
سیالکوٹ
گوجرانوالہ
گجرات
سرگودھا
بہاولپور
سکھر
لاڑکانہ
مردان
ایبٹ آباد
مانسہرہ
ہری پور
نوشہرہ
چارسدہ
صوابی
کوہاٹ
بنوں
ڈیرہ اسماعیل خان
ڈیرہ غازی خان
چترال
سوات
مینگورہ
گلگت
سکردو
ہنزہ
مظفرآباد
میرپور
جہلم
چکوال
اٹک
قصور
اوکاڑہ
ساہیوال
شیخوپورہ
جھنگ
ٹوبہ ٹیک سنگھ
رحیم یار خان
خانیوال
وہاڑی
لودھراں
خضدار
گوادر
تربت
حب
ژوب
زیارت
پنجاب
سندھ
بلوچستان
خیبر پختونخوا
کشمیر
""".strip().splitlines()


ORGANIZATIONS = """
اقوام متحدہ
عالمی ادارہ صحت
بین الاقوامی مالیاتی فنڈ
عالمی بینک
پاکستان تحریک انصاف
پاکستان مسلم لیگ ن
پاکستان پیپلز پارٹی
پاکستان کرکٹ بورڈ
پاکستان فوج
پاک فوج
سپریم کورٹ
الیکشن کمیشن
قومی اسمبلی
سینیٹ
وفاقی کابینہ
وزارت خارجہ
وزارت خزانہ
وزارت تعلیم
وزارت صحت
اسٹیٹ بینک
قومی احتساب بیورو
وفاقی تحقیقاتی ادارہ
پاکستان ٹیلی ویژن
ریڈیو پاکستان
ہائر ایجوکیشن کمیشن
نیشنل یونیورسٹی
قائد اعظم یونیورسٹی
اسلام آباد ہائی کورٹ
لاہور ہائی کورٹ
پشاور ہائی کورٹ
سندھ ہائی کورٹ
پاکستان ریلوے
پاکستان انٹرنیشنل ایئرلائنز
نیشنل ڈیٹا بیس
بی بی سی اردو
بی بی سی نیوز
ریسکیو 1122
انٹر سروسز پبلک ریلیشنز
پاکستان میڈیکل ایسوسی ایشن
""".strip().splitlines()


MISC_ENTITIES = """
کورونا وائرس
کووڈ
رمضان
عید الفطر
عید الاضحی
اولمپکس
ورلڈ کپ
چیمپئنز ٹرافی
ایشیا کپ
آئین پاکستان
""".strip().splitlines()


def build_gazetteer():
    """Create and validate entity dictionaries."""
    groups = {
        "PER": PERSONS,
        "LOC": LOCATIONS,
        "ORG": ORGANIZATIONS,
        "MISC": MISC_ENTITIES
    }

    gazetteer = {}

    for entity_type, names in groups.items():
        unique_names = {
            name.strip() for name in names if name.strip()
        }

        gazetteer[entity_type] = sorted(unique_names)

    assert len(gazetteer["PER"]) >= 50
    assert len(gazetteer["LOC"]) >= 50
    assert len(gazetteer["ORG"]) >= 30

    return gazetteer


class UrduNERTagger:
    """Longest-match gazetteer BIO tagger."""

    def __init__(self, gazetteer):
        self.gazetteer = gazetteer
        self.lookup = {}

        # Prefer longer phrases during matching.
        for entity_type, names in gazetteer.items():
            for name in names:
                tokens = tuple(name.split())
                self.lookup.setdefault(tokens, entity_type)

        self.max_length = max(
            len(name) for name in self.lookup
        )

    def annotate(self, tokens):
        tags = ["O"] * len(tokens)
        matches = []

        i = 0
        while i < len(tokens):
            found = None

            for length in range(
                min(self.max_length, len(tokens) - i),
                0,
                -1
            ):
                candidate = tuple(tokens[i:i + length])

                if candidate in self.lookup:
                    found = (
                        length,
                        self.lookup[candidate],
                        " ".join(candidate)
                    )
                    break

            if found is None:
                i += 1
                continue

            length, entity_type, entity_text = found

            tags[i] = f"B-{entity_type}"

            for j in range(i + 1, i + length):
                tags[j] = f"I-{entity_type}"

            matches.append({
                "start": i,
                "end": i + length,
                "type": entity_type,
                "text": entity_text
            })

            i += length

        return tags, matches


def annotate_splits(splits, tagger):
    """Apply BIO annotations to every sentence."""
    result = {}

    for split_name, sentences in splits.items():
        result[split_name] = []

        for sentence in sentences:
            tags, matches = tagger.annotate(
                sentence["tokens"]
            )

            assert len(tags) == len(sentence["tokens"])

            result[split_name].append({
                **sentence,
                "ner_tags": tags,
                "entity_matches": matches
            })

    return result


def export_annotations(annotated, output_dir):
    """Save CoNLL, JSON and review CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    review_path = output_dir / "ner_review.csv"

    with review_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as review_file:

        writer = csv.writer(review_file)
        writer.writerow([
            "split", "sentence_id", "article_id",
            "topic", "token_index", "token",
            "predicted_ner", "corrected_ner"
        ])

        for split_name, sentences in annotated.items():

            conll_path = output_dir / f"ner_{split_name}.conll"
            json_path = output_dir / f"ner_{split_name}.json"

            with conll_path.open(
                "w", encoding="utf-8"
            ) as conll_file:

                for sentence in sentences:
                    tokens = sentence["tokens"]
                    tags = sentence["ner_tags"]

                    for idx, (token, tag) in enumerate(
                        zip(tokens, tags)
                    ):
                        conll_file.write(
                            f"{token}\t{tag}\n"
                        )

                        writer.writerow([
                            split_name,
                            sentence["sentence_id"],
                            sentence["article_id"],
                            sentence["topic"],
                            idx,
                            token,
                            tag,
                            ""
                        ])

                    conll_file.write("\n")

            with json_path.open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(
                    sentences, f,
                    ensure_ascii=False, indent=2
                )


def audit_ner(annotated, gazetteer):
    """Summarize entity types and annotation coverage."""
    tag_counts = Counter()
    entity_counts = Counter()

    for sentences in annotated.values():
        for sentence in sentences:
            tag_counts.update(sentence["ner_tags"])

            for match in sentence["entity_matches"]:
                entity_counts[match["type"]] += 1

    return {
        "gazetteer_sizes": {
            label: len(names)
            for label, names in gazetteer.items()
        },
        "tag_distribution": dict(tag_counts),
        "entity_distribution": dict(entity_counts),
        "total_tokens": sum(tag_counts.values())
    }
