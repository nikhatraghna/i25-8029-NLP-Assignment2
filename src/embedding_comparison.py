"""Four-condition embedding comparison for DS5007."""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize


def prepare_vectors(matrix):
    """L2-normalize dense or sparse word representations."""
    if sparse.issparse(matrix):
        return normalize(matrix.tocsr(), norm="l2", axis=1)
    return normalize(
        np.asarray(matrix, dtype=np.float32),
        norm="l2", axis=1
    )


def similarity_scores(vectors, query_idx):
    """Cosine similarities against all vocabulary entries."""
    if sparse.issparse(vectors):
        scores = vectors @ vectors.getrow(query_idx).T
        return np.asarray(scores.toarray()).ravel()

    return np.asarray(
        vectors @ vectors[query_idx]
    ).ravel()


def rank_neighbors(vectors, vocab, query, top_k=5):
    """Find top-k neighbours, excluding query/special tokens."""
    if query not in vocab:
        return []

    scores = similarity_scores(vectors, vocab[query])
    idx2word = {idx: word for word, idx in vocab.items()}

    for word in ("<PAD>", "<UNK>", query):
        if word in vocab:
            scores[vocab[word]] = -np.inf

    ranking = np.argsort(-scores, kind="stable")[:top_k]

    return [
        (idx2word[int(idx)], float(scores[idx]))
        for idx in ranking
    ]


def evaluate_condition(matrix, vocab, pairs, queries, name):
    """Evaluate neighbours and MRR for one condition."""
    if matrix.shape[0] != len(vocab):
        raise ValueError(f"{name}: vocabulary/matrix mismatch")

    vectors = prepare_vectors(matrix)
    neighbor_rows = []
    mrr_rows = []

    for query in queries:
        for rank, (word, score) in enumerate(
            rank_neighbors(vectors, vocab, query), 1
        ):
            neighbor_rows.append({
                "condition": name,
                "query": query,
                "rank": rank,
                "neighbor": word,
                "similarity": score
            })

    for query, target in pairs:
        if query not in vocab or target not in vocab:
            mrr_rows.append({
                "condition": name,
                "query": query,
                "target": target,
                "rank": np.nan,
                "reciprocal_rank": 0.0,
                "status": "OOV"
            })
            continue

        scores = similarity_scores(vectors, vocab[query])

        for special in ("<PAD>", "<UNK>", query):
            if special in vocab:
                scores[vocab[special]] = -np.inf

        target_idx = vocab[target]
        target_score = scores[target_idx]

        # Deterministic ranking, including ties.
        indices = np.arange(len(scores))
        rank = 1 + int(np.sum(
            (scores > target_score) |
            ((scores == target_score) & (indices < target_idx))
        ))

        mrr_rows.append({
            "condition": name,
            "query": query,
            "target": target,
            "rank": rank,
            "reciprocal_rank": 1.0 / rank,
            "status": "OK"
        })

    return (
        pd.DataFrame(neighbor_rows),
        pd.DataFrame(mrr_rows)
    )
