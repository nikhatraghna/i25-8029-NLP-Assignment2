"""Custom PPMI implementation for BBC Urdu corpus."""

from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def build_cooccurrence(articles, vocab, window=5):
    """Build symmetric word-word co-occurrence counts."""
    size = len(vocab)
    unk = vocab["<UNK>"]
    matrix = csr_matrix((size, size), dtype=np.float32)

    for article_id in sorted(articles):
        rows, cols = [], []

        # Preserve sentence boundaries using corpus lines.
        for sentence in articles[article_id].splitlines():
            tokens = sentence.split()
            ids = [vocab.get(token, unk) for token in tokens]

            for i, centre in enumerate(ids):
                end = min(len(ids), i + window + 1)

                for j in range(i + 1, end):
                    context = ids[j]

                    rows.extend((centre, context))
                    cols.extend((context, centre))

        if rows:
            values = np.ones(len(rows), dtype=np.float32)
            block = coo_matrix(
                (values, (rows, cols)),
                shape=(size, size),
                dtype=np.float32
            ).tocsr()

            matrix += block

    matrix.sum_duplicates()
    return matrix


def compute_ppmi(cooccurrence):
    """Compute max(0, log2(P(w,c)/(P(w)P(c))))."""
    counts = cooccurrence.tocoo()
    total = float(counts.data.sum())

    if total <= 0:
        raise ValueError("Empty co-occurrence matrix.")

    row_sum = np.asarray(
        cooccurrence.sum(axis=1)
    ).ravel()

    col_sum = np.asarray(
        cooccurrence.sum(axis=0)
    ).ravel()

    numerator = counts.data.astype(np.float64) * total

    denominator = (
        row_sum[counts.row].astype(np.float64)
        * col_sum[counts.col].astype(np.float64)
    )

    values = np.maximum(
        np.log2(numerator / denominator), 0.0
    ).astype(np.float32)

    ppmi = csr_matrix(
        (values, (counts.row, counts.col)),
        shape=cooccurrence.shape
    )

    ppmi.eliminate_zeros()
    return ppmi


def nearest_neighbours(ppmi, vocab, query, top_k=5):
    """Return the closest words using cosine similarity."""
    if query not in vocab:
        return []

    query_idx = vocab[query]

    scores = cosine_similarity(
        ppmi[query_idx],
        ppmi
    ).ravel()

    idx2word = {
        idx: word for word, idx in vocab.items()
    }

    ranked = np.argsort(scores)[::-1]
    results = []

    for idx in ranked:
        word = idx2word[idx]

        if word in ("<UNK>", "<PAD>") or word == query:
            continue

        results.append((word, float(scores[idx])))

        if len(results) == top_k:
            break

    return results


def save_ppmi(ppmi, output_path, block_size=256):
    """Save dense .npy output without a large RAM spike."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=ppmi.shape
    )

    for start in range(0, ppmi.shape[0], block_size):
        end = min(start + block_size, ppmi.shape[0])
        output[start:end] = ppmi[start:end].toarray()

    output.flush()
    del output
