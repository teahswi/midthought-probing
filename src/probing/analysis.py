"""Baseline and result-audit helpers."""

import json
from pathlib import Path


def analyze_baselines(per_example_meta, probe_results, fig_path=None, results_path=None):
    correct = sum(int(row.get("label", 0)) for row in per_example_meta)
    total = len(per_example_meta)
    record = {
        "n_examples": total,
        "correct": correct,
        "incorrect": total - correct,
        "prevalence": correct / total if total else None,
        "probe_results": probe_results,
    }
    if results_path:
        path = Path(results_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def audit_result_metadata(results_dir: str | Path):
    """Mark result files lacking problem IDs as unvalidated, never as generalization evidence."""
    directory = Path(results_dir)
    records = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        has_ids = bool(payload.get("problem_ids") or payload.get("example_ids"))
        records.append(
            {
                "file": str(path),
                "has_problem_ids": has_ids,
                "validation_status": "validated" if has_ids else "unvalidated",
            }
        )
    return records
