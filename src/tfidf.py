"""TF-IDF implementation for DS5007 Assignment 2."""

from collections import Counter
from pathlib import Path
import json

import numpy as np
from scipy.sparse import csr_matrix


def build_tfidf_vocab(articles, max_words=10000):
    """Keep the most frequent 10,000 corpus tokens."""
    counts = Counter()

    for article in articles.values():
        counts.update(article.split())

    words = [
        word for word, _ in counts.most_common()
        if word != "<UNK>"
    ][:max_words]

    vocab = {"<UNK>": 0}
    vocab.update({
        word: idx + 1
        for idx, word in enumerate(words)
    })

    return vocab


def build_term_document_matrix(articles, vocab):
    """Construct a document-by-term frequency matrix."""
    article_ids = sorted(articles)
    rows, cols, values = [], [], []

    for row_idx, article_id in enumerate(article_ids):
        token_counts = Counter(
            vocab.get(token, vocab["<UNK>"])
            for token in articles[article_id].split()
        )

        for token_idx, count in token_counts.items():
            rows.append(row_idx)
            cols.append(token_idx)
            values.append(count)

    matrix = csr_matrix(
        (values, (rows, cols)),
        shape=(len(article_ids), len(vocab)),
        dtype=np.float32
    )

    return matrix, article_ids


def compute_tfidf(tf_matrix):
    """Apply the TF-IDF formula specified by the instructor."""
    n_docs = tf_matrix.shape[0]

    document_frequency = np.asarray(
        (tf_matrix > 0).sum(axis=0)
    ).ravel()

    idf = np.log(
        n_docs / (1.0 + document_frequency)
    )

    tfidf = tf_matrix.multiply(idf).tocsr()

    return tfidf, idf


def top_topic_words(tfidf, article_ids, labels, vocab, top_n=10):
    """Return top TF-IDF words for each labelled topic."""
    idx2word = {
        index: word for word, index in vocab.items()
    }

    results = {}
    topics = sorted(set(labels.values()))

    for topic in topics:
        row_indices = [
            idx for idx, article_id in enumerate(article_ids)
            if labels.get(article_id) == topic
        ]

        if not row_indices:
            continue

        scores = np.asarray(
            tfidf[row_indices].mean(axis=0)
        ).ravel()

        ranked = np.argsort(scores)[::-1]

        results[topic] = [
            (idx2word[idx], float(scores[idx]))
            for idx in ranked
            if idx2word[idx] != "<UNK>"
        ][:top_n]

    return results


def save_tfidf(tfidf, vocab, output_dir):
    """Save matrix and vocabulary mapping."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        output_dir / "tfidf_matrix.npy",
        tfidf.toarray().astype(np.float32)
    )

    with open(
        output_dir / "tfidf_word2idx.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
