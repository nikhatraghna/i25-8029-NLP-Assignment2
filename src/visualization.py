"""Topic-aware PPMI t-SNE visualization for DS5007 Assignment 2."""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE


TOPICS = [
    "Politics",
    "Sports",
    "Economy",
    "International",
    "Health & Society"
]


def build_word_topic_map(
    articles,
    labeled_metadata,
    min_count=3,
    min_lift=1.5
):
    """
    Assign each word to a topic using relative frequency and lift.

    Words without sufficiently strong topic association
    are assigned to 'Other'.
    """

    topic_word_counts = defaultdict(Counter)
    topic_totals = Counter()
    global_counts = Counter()

    for record in labeled_metadata:
        article_id = int(record["article_number"])

        if article_id not in articles:
            raise ValueError(
                f"Missing cleaned article: {article_id}"
            )

        # Prefer reviewed labels when explicitly supplied.
        verified = record.get("verified_topic")
        topic = verified or record.get("auto_topic")

        # Exclude uncertain automatic labels unless reviewed.
        if not verified and record.get("needs_review", False):
            continue

        if topic not in TOPICS:
            continue

        words = articles[article_id].split()

        topic_word_counts[topic].update(words)
        topic_totals[topic] += len(words)
        global_counts.update(words)

    total_tokens = sum(topic_totals.values())

    if total_tokens == 0:
        raise ValueError("No usable labeled articles found.")

    word_categories = {}

    for word, total_count in global_counts.items():

        if total_count < min_count:
            word_categories[word] = "Other"
            continue

        global_prob = total_count / total_tokens

        scores = {}

        for topic in TOPICS:
            if topic_totals[topic] == 0:
                continue

            topic_prob = (
                topic_word_counts[topic][word]
                / topic_totals[topic]
            )

            scores[topic] = topic_prob / global_prob

        if not scores:
            word_categories[word] = "Other"
            continue

        best_topic = max(scores, key=scores.get)
        best_lift = scores[best_topic]

        if best_lift >= min_lift:
            word_categories[word] = best_topic
        else:
            word_categories[word] = "Other"

    return word_categories


def plot_ppmi_tsne(
    ppmi,
    vocab,
    frequencies,
    output_path,
    word_categories,
    top_n=200,
    seed=42
):
    """
    Plot PPMI representations of the 200 most frequent words.
    """

    words = [
        word
        for word, _ in frequencies.most_common()
        if word in vocab
        and word not in ("<PAD>", "<UNK>")
    ][:top_n]

    if len(words) < top_n:
        raise ValueError(
            f"Only {len(words)} words available; "
            f"{top_n} requested."
        )

    indices = [vocab[word] for word in words]

    # Select sparse PPMI representations.
    vectors = ppmi[indices]

    # Dimensionality reduction before t-SNE.
    svd = TruncatedSVD(
        n_components=min(50, top_n - 1),
        random_state=seed
    )

    reduced = svd.fit_transform(vectors)

    tsne = TSNE(
        n_components=2,
        perplexity=20,
        learning_rate="auto",
        init="pca",
        random_state=seed
    )

    coordinates = tsne.fit_transform(reduced)

    categories = [
        word_categories.get(word, "Other")
        for word in words
    ]

    fig, ax = plt.subplots(figsize=(18, 13))

    # Plot each semantic category.
    for topic in TOPICS + ["Other"]:

        positions = [
            idx
            for idx, category in enumerate(categories)
            if category == topic
        ]

        if not positions:
            continue

        ax.scatter(
            coordinates[positions, 0],
            coordinates[positions, 1],
            s=35,
            alpha=0.8,
            label=f"{topic} ({len(positions)})"
        )

    # Display word labels.
    for idx, word in enumerate(words):
        ax.annotate(
            word,
            (
                coordinates[idx, 0],
                coordinates[idx, 1]
            ),
            fontsize=7,
            alpha=0.8
        )

    ax.set_title(
        "PPMI Word Embeddings: t-SNE of 200 Frequent Urdu Tokens",
        fontsize=15
    )

    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")

    ax.legend(
        title="Semantic Category",
        loc="best"
    )

    ax.grid(alpha=0.15)

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    return words, categories, coordinates
