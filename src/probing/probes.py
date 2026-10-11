"""Logistic-regression latent correctness probes."""

import random
import numpy as np


class ProbeResults(dict):
    """Dictionary subclass storing probe evaluation results with preserved split metadata."""

    def __init__(self, *args, split_manifest: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_manifest = split_manifest or {}


def _sigmoid(z):
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _compute_auroc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1)
    unique_probs, inverse, counts = np.unique(y_prob, return_inverse=True, return_counts=True)
    if len(unique_probs) < len(y_prob):
        for i, count in enumerate(counts):
            if count > 1:
                tied_indices = np.where(inverse == i)[0]
                ranks[tied_indices] = np.mean(ranks[tied_indices])
    rank_sum_pos = np.sum(ranks[pos_idx])
    u = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


def _fit_numpy_logistic_regression(X_train, y_train, X_test, seed: int = 42, max_iter: int = 500, lr: float = 0.1, reg: float = 1e-3):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0.0] = 1.0
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    n_samples, n_features = X_train_scaled.shape
    weights = np.zeros(n_features)
    bias = 0.0

    n_pos = max(1, np.sum(y_train == 1))
    n_neg = max(1, np.sum(y_train == 0))
    w_pos = n_samples / (2.0 * n_pos)
    w_neg = n_samples / (2.0 * n_neg)
    sample_weights = np.where(y_train == 1, w_pos, w_neg)

    for _ in range(max_iter):
        preds = _sigmoid(np.dot(X_train_scaled, weights) + bias)
        errors = (preds - y_train) * sample_weights
        grad_w = np.dot(X_train_scaled.T, errors) / n_samples + reg * weights
        grad_b = np.sum(errors) / n_samples
        weights -= lr * grad_w
        bias -= lr * grad_b

    return _sigmoid(np.dot(X_test_scaled, weights) + bias)


def _compute_metrics_numpy(y_test, probability, y_train=None):
    y_test_arr = np.asarray(y_test, dtype=int)
    predicted = (probability >= 0.5).astype(int)

    tp = int(np.sum((predicted == 1) & (y_test_arr == 1)))
    fp = int(np.sum((predicted == 1) & (y_test_arr == 0)))
    tn = int(np.sum((predicted == 0) & (y_test_arr == 0)))
    fn = int(np.sum((predicted == 0) & (y_test_arr == 1)))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = 0.5 * (tpr + tnr)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tpr
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    brier = float(np.mean((probability - y_test_arr) ** 2))
    auroc = _compute_auroc(y_test_arr, probability) if len(set(y_test_arr)) > 1 else None

    maj_acc = max(float(np.mean(y_test_arr)), 1.0 - float(np.mean(y_test_arr))) if len(y_test_arr) > 0 else 0.5

    return {
        "auroc": auroc,
        "balanced_accuracy": float(balanced_acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "brier_score": brier,
        "probabilities": probability.tolist(),
        "majority_acc": maj_acc,
        "majority_baseline_accuracy": maj_acc,
    }


def run_probe(X_train, y_train, X_test, y_test, seed: int = 42):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            balanced_accuracy_score,
            brier_score_loss,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        train = scaler.fit_transform(np.asarray(X_train))
        test = scaler.transform(np.asarray(X_test))
        classifier = LogisticRegression(
            class_weight="balanced", max_iter=2000, solver="liblinear", random_state=seed
        )
        classifier.fit(train, y_train)
        probability = classifier.predict_proba(test)[:, 1]
        predicted = (probability >= 0.5).astype(int)

        y_test_arr = np.asarray(y_test)
        majority_class = int(np.round(np.mean(y_train))) if len(y_train) > 0 else 0
        majority_acc = float(np.mean(y_test_arr == majority_class)) if len(y_test_arr) > 0 else 0.5

        return {
            "auroc": float(roc_auc_score(y_test, probability)) if len(set(y_test)) > 1 else None,
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
            "precision": float(precision_score(y_test, predicted, zero_division=0)),
            "recall": float(recall_score(y_test, predicted, zero_division=0)),
            "f1": float(f1_score(y_test, predicted, zero_division=0)),
            "brier_score": float(brier_score_loss(y_test, probability)),
            "probabilities": probability.tolist(),
            "majority_acc": majority_acc,
            "majority_baseline_accuracy": max(float(np.mean(y_test_arr)), 1.0 - float(np.mean(y_test_arr))) if len(y_test_arr) > 0 else 0.5,
        }
    except (ImportError, OSError):
        # Graceful pure-NumPy fallback when sklearn binary extension is restricted
        probability = _fit_numpy_logistic_regression(X_train, y_train, X_test, seed=seed)
        return _compute_metrics_numpy(y_test, probability, y_train=y_train)


def run_all_probes(
    features_by_t,
    labels_by_t,
    seed: int = 42,
    example_ids: list[str] | None = None,
    train_ratio: float = 0.8,
    train_indices: list[int] | None = None,
    test_indices: list[int] | None = None,
    metadata_lengths: list[int] | None = None,
    return_manifest: bool = False,
):
    """Train one clearly specified probe at each reasoning checkpoint using a randomized or ID-based split."""
    from .data import split_ids

    # Determine total sample count from the first available checkpoint
    first_step = next(iter(labels_by_t)) if labels_by_t else None
    n_samples = len(labels_by_t[first_step]) if first_step is not None else 0

    if train_indices is not None and test_indices is not None:
        sel_train_idx = list(train_indices)
        sel_test_idx = list(test_indices)
    elif example_ids is not None and len(example_ids) == n_samples:
        unique_ids = list(dict.fromkeys(example_ids))
        train_ids, test_ids = split_ids(unique_ids, train_ratio=train_ratio, seed=seed)
        train_id_set = set(train_ids)
        sel_train_idx = [i for i, eid in enumerate(example_ids) if eid in train_id_set]
        sel_test_idx = [i for i, eid in enumerate(example_ids) if eid not in train_id_set]
    else:
        indices = list(range(n_samples))
        random.Random(seed).shuffle(indices)
        cutoff = max(1, int(n_samples * train_ratio)) if n_samples > 1 else 1
        sel_train_idx = indices[:cutoff]
        sel_test_idx = indices[cutoff:]

    split_manifest = {
        "seed": seed,
        "train_ratio": train_ratio,
        "train_indices": sel_train_idx,
        "test_indices": sel_test_idx,
        "n_train": len(sel_train_idx),
        "n_test": len(sel_test_idx),
    }
    if example_ids is not None and len(example_ids) == n_samples:
        split_manifest["train_ids"] = [example_ids[i] for i in sel_train_idx]
        split_manifest["test_ids"] = [example_ids[i] for i in sel_test_idx]

    results = ProbeResults(split_manifest=split_manifest)

    # Optional length metadata baseline (non-hidden-state baseline)
    if metadata_lengths is not None and len(metadata_lengths) == n_samples and sel_test_idx:
        first_labels = np.asarray(labels_by_t[first_step])
        len_train = np.asarray(metadata_lengths, dtype=float)[sel_train_idx].reshape(-1, 1)
        len_test = np.asarray(metadata_lengths, dtype=float)[sel_test_idx].reshape(-1, 1)
        y_train_len = first_labels[sel_train_idx]
        y_test_len = first_labels[sel_test_idx]
        if len(set(y_train_len)) > 1 and len(set(y_test_len)) > 1:
            split_manifest["metadata_length_baseline"] = run_probe(
                len_train, y_train_len, len_test, y_test_len, seed=seed
            )

    for step, features in features_by_t.items():
        labels = np.asarray(labels_by_t[step])
        if len(features) == 0 or len(labels) == 0:
            results[step] = {"auroc": None, "reason": "empty features or labels"}
            continue

        X_train = np.asarray(features)[sel_train_idx]
        y_train = labels[sel_train_idx]
        X_test = np.asarray(features)[sel_test_idx]
        y_test = labels[sel_test_idx]

        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
            majority_acc = (
                float(np.mean(y_test == int(np.round(np.mean(y_train)))))
                if len(y_test) > 0 and len(y_train) > 0
                else 0.5
            )
            results[step] = {
                "auroc": None,
                "reason": "insufficient class diversity",
                "n_train": len(y_train),
                "n_test": len(y_test),
                "majority_acc": majority_acc,
            }
            continue

        probe_eval = run_probe(X_train, y_train, X_test, y_test, seed)
        probe_eval["n_train"] = len(y_train)
        probe_eval["n_test"] = len(y_test)
        results[step] = probe_eval

    if return_manifest:
        return results, split_manifest
    return results
