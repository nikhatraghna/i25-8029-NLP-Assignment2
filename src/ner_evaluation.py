"""Entity-level BIO NER evaluation.

Exact-match entities are compared by sentence, start/end
token positions and entity type. No external NER library.
"""

from collections import Counter


def extract_entities(tags):
    """Return (entity_type, start, end_exclusive) spans."""
    entities = []
    active_type = None
    start = None

    def close(end):
        nonlocal active_type, start
        if active_type is not None:
            entities.append((active_type, start, end))
        active_type = None
        start = None

    for i, tag in enumerate(list(tags) + ["O"]):
        if tag == "O":
            close(i)
            continue

        if "-" not in tag:
            raise ValueError(f"Invalid BIO tag: {tag}")

        prefix, entity_type = tag.split("-", 1)

        if prefix == "B":
            close(i)
            active_type = entity_type
            start = i

        elif prefix == "I":
            if active_type != entity_type:
                # Recovery convention for invalid predicted BIO:
                # treat an orphan I-tag as the beginning of a span.
                close(i)
                active_type = entity_type
                start = i
        else:
            raise ValueError(f"Invalid BIO prefix: {prefix}")

    return entities


def entity_metrics(y_true, y_pred):
    """Micro entity precision, recall, F1 and per-type metrics."""
    if len(y_true) != len(y_pred):
        raise ValueError("Sentence counts must match")

    tp = Counter()
    fp = Counter()
    fn = Counter()

    errors_fp = []
    errors_fn = []

    for sentence_index, (gold, pred) in enumerate(
        zip(y_true, y_pred)
    ):
        if len(gold) != len(pred):
            raise ValueError(
                f"Tag length mismatch in sentence {sentence_index}"
            )

        gold_entities = set(extract_entities(gold))
        pred_entities = set(extract_entities(pred))

        for entity in gold_entities & pred_entities:
            tp[entity[0]] += 1

        for entity in pred_entities - gold_entities:
            fp[entity[0]] += 1
            errors_fp.append((sentence_index, entity))

        for entity in gold_entities - pred_entities:
            fn[entity[0]] += 1
            errors_fn.append((sentence_index, entity))

    def scores(t, f_p, f_n):
        precision = t / (t + f_p) if t + f_p else 0.0
        recall = t / (t + f_n) if t + f_n else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": t,
            "fp": f_p,
            "fn": f_n,
        }

    result = scores(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    result["per_type"] = {
        tag: scores(tp[tag], fp[tag], fn[tag])
        for tag in ("PER", "LOC", "ORG", "MISC")
    }
    result["false_positives"] = errors_fp
    result["false_negatives"] = errors_fn

    return result


def evaluate_ner_model(model, loader, tag_names, device):
    """Evaluate complete entities on a sequence-tagging DataLoader."""
    model.eval()
    all_gold = []
    all_pred = []

    import torch

    with torch.no_grad():
        for batch in loader:
            x = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)

            predicted = model.predict(x, mask)

            for i, path in enumerate(predicted):
                n = int(mask[i].sum().item())

                gold_ids = batch["tags"][i, :n].tolist()

                if len(path) != n:
                    raise ValueError("Prediction length mismatch")

                all_gold.append([tag_names[j] for j in gold_ids])
                all_pred.append([tag_names[j] for j in path])

    return entity_metrics(all_gold, all_pred)
