"""Word embedding evaluation for DS5007 Assignment 2."""

import numpy as np
import pandas as pd


def normalize_embeddings(embeddings):
    """L2-normalize embedding vectors for cosine similarity."""
    embeddings = np.asarray(embeddings, dtype=np.float32)

    norms = np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )

    return embeddings / np.maximum(norms, 1e-12)


def nearest_neighbors(
    embeddings, word2idx, query, top_k=10
):
    """Retrieve nearest words using cosine similarity."""
    if query not in word2idx:
        return []

    normalized = normalize_embeddings(embeddings)
    idx2word = {
        idx: word for word, idx in word2idx.items()
    }

    query_idx = word2idx[query]
    scores = normalized @ normalized[query_idx]

    excluded = {"<PAD>", "<UNK>", query}
    ranked = np.argsort(-scores, kind="stable")

    results = []

    for idx in ranked:
        word = idx2word.get(int(idx))
        if word is None or word in excluded:
            continue

        results.append({
            "query": query,
            "neighbor": word,
            "similarity": float(scores[idx]),
            "rank": len(results) + 1
        })

        if len(results) >= top_k:
            break

    return results


def word_analogy(
    embeddings, word2idx, a, b, c, top_k=3
):
    """Calculate v(b) - v(a) + v(c)."""
    words = [a, b, c]

    missing = [w for w in words if w not in word2idx]
    if missing:
        return {"missing": missing, "predictions": []}

    normalized = normalize_embeddings(embeddings)

    vector = (
        normalized[word2idx[b]]
        - normalized[word2idx[a]]
        + normalized[word2idx[c]]
    )

    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return {"missing": [], "predictions": []}

    vector = vector / norm
    scores = normalized @ vector

    idx2word = {
        idx: word for word, idx in word2idx.items()
    }

    excluded = set(words) | {"<PAD>", "<UNK>"}
    ranked = np.argsort(-scores, kind="stable")

    predictions = []

    for idx in ranked:
        word = idx2word.get(int(idx))

        if word is None or word in excluded:
            continue

        predictions.append({
            "word": word,
            "similarity": float(scores[idx])
        })

        if len(predictions) == top_k:
            break

    return {"missing": [], "predictions": predictions}


def evaluate_analogies(embeddings, word2idx, tests):
    """Evaluate analogy tests against manually specified answers."""
    records = []

    for a, b, c, expected in tests:
        result = word_analogy(
            embeddings, word2idx, a, b, c, top_k=3
        )

        predictions = [
            item["word"] for item in result["predictions"]
        ]

        missing = result["missing"]
        if expected not in word2idx:
            missing = list(dict.fromkeys(missing + [expected]))

        records.append({
            "a": a,
            "b": b,
            "c": c,
            "expected": expected,
            "top_1": predictions[0] if len(predictions) > 0 else None,
            "top_2": predictions[1] if len(predictions) > 1 else None,
            "top_3": predictions[2] if len(predictions) > 2 else None,
            "correct_top_3": (
                expected in predictions if not missing else False
            ),
            "missing_words": ", ".join(missing)
        })

    return pd.DataFrame(records)


def calculate_mrr(embeddings, word2idx, labeled_pairs):
    """
    Mean Reciprocal Rank for manually labeled
    directional query-target word pairs.
    """
    normalized = normalize_embeddings(embeddings)
    ranks = []

    for query, target in labeled_pairs:
        if query not in word2idx or target not in word2idx:
            ranks.append({
                "query": query,
                "target": target,
                "rank": None,
                "reciprocal_rank": 0.0,
                "status": "OOV"
            })
            continue

        query_idx = word2idx[query]
        target_idx = word2idx[target]

        if query_idx == target_idx:
            raise ValueError("Query and target must differ.")

        scores = normalized @ normalized[query_idx]

        valid = np.ones(len(scores), dtype=bool)
        valid[query_idx] = False

        for special in ("<PAD>", "<UNK>"):
            if special in word2idx:
                valid[word2idx[special]] = False

        target_score = scores[target_idx]
        valid_indices = np.flatnonzero(valid)

        # Rank with deterministic tie-breaking by index.
        rank = 1 + sum(
            scores[i] > target_score or
            (scores[i] == target_score and i < target_idx)
            for i in valid_indices
        )

        ranks.append({
            "query": query,
            "target": target,
            "rank": int(rank),
            "reciprocal_rank": 1.0 / rank,
            "status": "OK"
        })

    table = pd.DataFrame(ranks)

    mrr = (
        float(table["reciprocal_rank"].mean())
        if len(table) else 0.0
    )

    return mrr, table
