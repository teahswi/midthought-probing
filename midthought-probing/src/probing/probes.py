"""Logistic-regression latent correctness probes."""

import numpy as np


def run_probe(X_train, y_train, X_test, y_test, seed: int = 42):
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
    return {
        "auroc": float(roc_auc_score(y_test, probability)) if len(set(y_test)) > 1 else None,
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(y_test, probability)),
        "probabilities": probability.tolist(),
    }


def run_all_probes(features_by_t, labels_by_t, seed: int = 42):
    """Train one clearly specified probe at each reasoning checkpoint."""
    results = {}
    for step, features in features_by_t.items():
        labels = np.asarray(labels_by_t[step])
        midpoint = max(1, int(len(labels) * 0.8))
        if len(set(labels[:midpoint])) < 2 or len(set(labels[midpoint:])) < 2:
            results[step] = {"auroc": None, "reason": "insufficient class diversity"}
            continue
        results[step] = run_probe(
            features[:midpoint], labels[:midpoint], features[midpoint:], labels[midpoint:], seed
        )
    return results
