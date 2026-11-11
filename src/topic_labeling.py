"""Rule-based weak supervision for BBC Urdu topic labels."""

import csv
import json
import re
from collections import Counter
from pathlib import Path


TOPIC_KEYWORDS = {
    "Politics": [
        "حکومت", "حکومتی", "سیاست", "سیاسی",
        "وزیر", "انتخاب", "الیکشن", "پارلیمان",
        "اسمبلی", "پارٹی", "عمران خان",
        "صدر", "قانون", "احتجاج"
    ],
    "Sports": [
        "کرکٹ", "میچ", "ٹیم", "کھلاڑی",
        "کپتان", "وکٹ", "رن", "فٹبال",
        "ٹورنامنٹ", "کھیل", "اسکور",
        "بیٹنگ", "پی سی بی"
    ],
    "Economy": [
        "معیشت", "معاشی", "تجارت", "بینک",
        "مہنگائی", "بجٹ", "سرمایہ", "روپے",
        "ڈالر", "تیل", "قیمت", "ٹیکس",
        "کاروبار", "برآمد", "درآمد"
    ],
    "International": [
        "اقوام متحدہ", "امریکہ", "چین",
        "روس", "اسرائیل", "ایران",
        "افغانستان", "سعودی عرب",
        "خارجہ", "عالمی", "بین الاقوامی",
        "جنگ", "سفارتی", "معاہدہ"
    ],
    "Health & Society": [
        "صحت", "ہسپتال", "بیماری", "علاج",
        "ڈاکٹر", "مریض", "ویکسین",
        "تعلیم", "سکول", "اسکول",
        "بچہ", "بچے", "خواتین",
        "سیلاب", "معاشرہ", "طبی"
    ]
}


def count_keyword(text, keyword):
    """Count whole-word or whole-phrase matches."""
    pattern = (
        r"(?<!\w)" +
        re.escape(keyword).replace(r"\ ", r"\s+") +
        r"(?!\w)"
    )
    return len(re.findall(pattern, text))


def score_article(title, body):
    """Combine title and document keyword evidence."""
    scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        title_score = sum(
            count_keyword(title, keyword)
            for keyword in keywords
        )

        body_score = sum(
            min(count_keyword(body, keyword), 3)
            for keyword in keywords
        )

        scores[topic] = 3 * title_score + body_score

    return scores


def classify_article(title, body):
    scores = score_article(title, body)
    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0])
    )

    best_topic, best_score = ranked[0]
    second_score = ranked[1][1]

    # Do not invent a confident label without evidence.
    needs_review = (
        best_score < 2 or
        best_score - second_score < 2
    )

    return {
        "auto_topic": best_topic if best_score > 0 else None,
        "topic_scores": scores,
        "needs_review": needs_review,
        "score_margin": best_score - second_score
    }


def generate_labels(metadata, cleaned):
    labeled = []

    for record in metadata:
        article_id = int(record["article_number"])

        if article_id not in cleaned:
            raise ValueError(
                f"Article {article_id} missing from cleaned corpus"
            )

        result = classify_article(
            record.get("title", ""),
            cleaned[article_id]
        )

        labeled.append({**record, **result})

    return labeled


def save_labels(labeled, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(
        output_dir / "Metadata_labeled.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(labeled, f, ensure_ascii=False, indent=2)

    with open(
        output_dir / "topic_review.csv",
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "article_number", "title", "auto_topic",
            "score_margin", "needs_review",
            "verified_topic"
        ])

        for record in labeled:
            writer.writerow([
                record["article_number"],
                record["title"],
                record["auto_topic"],
                record["score_margin"],
                record["needs_review"],
                ""
            ])
