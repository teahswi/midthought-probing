"""Baseline and result-audit helpers."""

import json
from pathlib import Path


def analyze_baselines(per_example_meta, probe_results, fig_path=None, results_path=None):
    correct = sum(int(row.get("label", 0)) for row in per_example_meta)
    total = len(per_example_meta)
    prevalence = correct / total if total else None
    majority_baseline = max(prevalence, 1.0 - prevalence) if prevalence is not None else 0.5

    # Early vs late comparison
    valid_aurocs = {
        int(step): res.get("auroc")
        for step, res in probe_results.items()
        if isinstance(res, dict) and res.get("auroc") is not None
    }
    sorted_steps = sorted(valid_aurocs.keys())
    early_auroc = valid_aurocs[sorted_steps[0]] if sorted_steps else None
    late_auroc = valid_aurocs[sorted_steps[-1]] if sorted_steps else None
    auroc_gain = (
        (late_auroc - early_auroc)
        if (early_auroc is not None and late_auroc is not None)
        else None
    )

    # Length baseline availability check
    lengths = [
        row.get("reasoning_token_count")
        for row in per_example_meta
        if "reasoning_token_count" in row
    ]
    has_lengths = len(lengths) == total and total > 0

    record = {
        "n_examples": total,
        "correct": correct,
        "incorrect": total - correct,
        "prevalence": prevalence,
        "majority_baseline_accuracy": majority_baseline,
        "early_checkpoint_auroc": early_auroc,
        "late_checkpoint_auroc": late_auroc,
        "auroc_gain_early_to_late": auroc_gain,
        "has_length_metadata": has_lengths,
        "probe_results": probe_results,
    }
    if results_path:
        path = Path(results_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def audit_result_metadata(results_dir: str | Path):
    """Mark result files lacking problem IDs or split manifests as unvalidated, never as generalization evidence."""
    directory = Path(results_dir)
    records = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        has_ids = bool(
            payload.get("problem_ids")
            or payload.get("example_ids")
            or payload.get("split_manifest")
        )
        records.append(
            {
                "file": str(path),
                "has_problem_ids": has_ids,
                "validation_status": "validated" if has_ids else "unvalidated",
            }
        )
    return records
