import csv
from pathlib import Path

from src.common import numeric_correctness, parse_numeric_answer, select_snapshot_positions


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_parse_numeric_answer_handles_common_formats():
    cases = {
        '42': 42.0,
        '$1,234.50': 1234.5,
        '3/2': 1.5,
        '-17.2': -17.2,
        '\\boxed{8}': 8.0,
    }
    for raw, expected in cases.items():
        value = parse_numeric_answer(raw)
        assert value is not None
        assert abs(value - expected) < 1e-6


def test_numeric_correctness_rejects_unparseable_cases():
    assert numeric_correctness('12', '12.0') is True
    assert numeric_correctness('12', '13') is False
    assert numeric_correctness('abc', '12') is None


def test_select_snapshot_positions_uses_reasoning_progress():
    positions = select_snapshot_positions(100, [10, 50, 90])
    assert positions == [10, 50, 90]


def test_plot_generation_from_synthetic_fixture(tmp_path):
    write_csv(tmp_path / 'metrics.csv', ['progress', 'auroc', 'balanced_accuracy', 'f1'], [
        {'progress': 10, 'auroc': 0.61, 'balanced_accuracy': 0.58, 'f1': 0.5},
        {'progress': 50, 'auroc': 0.72, 'balanced_accuracy': 0.66, 'f1': 0.6},
    ])
    write_csv(tmp_path / 'predictions.csv', ['sample_id', 'true_label', 'predicted_probability'], [
        {'sample_id': 'a', 'true_label': 0, 'predicted_probability': 0.2},
        {'sample_id': 'b', 'true_label': 1, 'predicted_probability': 0.8},
    ])

    import subprocess
    subprocess.run([
        'python', str(Path(__file__).resolve().parents[1] / 'src' / 'plot_results.py'),
        '--metrics-csv', str(tmp_path / 'metrics.csv'),
        '--predictions-csv', str(tmp_path / 'predictions.csv'),
        '--output-dir', str(tmp_path / 'figures'),
        '--dpi', '120',
    ], check=True)
    assert (tmp_path / 'figures' / 'auroc_vs_reasoning_progress.png').exists()
    assert (tmp_path / 'figures' / 'reliability_diagram.png').exists()


def test_main_cli_preserves_config_path(tmp_path, monkeypatch, capsys):
    from main import run_legacy
    dummy_yaml = tmp_path / "custom_experiment.yaml"
    dummy_yaml.write_text("model: {id: test}\n", encoding="utf-8")

    # In dry-run mode, run_legacy should echo the command containing the custom config path
    result = run_legacy("load_dataset.py", str(dummy_yaml), dry_run=True)
    assert result == 0
    captured = capsys.readouterr().out
    assert str(dummy_yaml.resolve()) in captured
    assert "experiment.json" not in captured


def test_randomized_train_test_split_by_question_id():
    from src.probing.probes import run_all_probes
    import numpy as np

    # 10 examples with question IDs, 5 class 1, 5 class 0
    example_ids = [f"q_{i}" for i in range(10)]
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    # Synthetic features: 4 features each
    features_step_10 = np.random.RandomState(42).randn(10, 4)
    features_step_50 = np.random.RandomState(43).randn(10, 4)
    features_by_t = {10: features_step_10, 50: features_step_50}
    labels_by_t = {10: labels, 50: labels}

    results, manifest = run_all_probes(
        features_by_t, labels_by_t, seed=42, example_ids=example_ids, train_ratio=0.8, return_manifest=True
    )

    train_ids = set(manifest["train_ids"])
    test_ids = set(manifest["test_ids"])
    assert len(train_ids) > 0 and len(test_ids) > 0
    # Crucial scientific safeguard: train and test questions must be strictly disjoint
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(test_ids) == 10
    # Split manifest must record the seed and ratios
    assert manifest["seed"] == 42
    assert manifest["train_ratio"] == 0.8
    # Shuffling with a different seed produces a different split
    _, manifest_other = run_all_probes(
        features_by_t, labels_by_t, seed=999, example_ids=example_ids, train_ratio=0.8, return_manifest=True
    )
    assert manifest["train_indices"] != manifest_other["train_indices"]


def test_answer_leakage_safeguard():
    from src.probing.generation import find_reasoning_boundary, split_reasoning_prefix
    from src.probing.features import validate_answer_free_prefix

    response = "Step 1: Calculate 5 * 6 = 30. Step 2: Add 12 to get 42. Therefore, the final answer is \\boxed{42}."
    boundary = find_reasoning_boundary(response)
    reasoning, answer = split_reasoning_prefix(response)

    assert "\\boxed{" not in reasoning
    assert "42" in answer
    assert "Therefore, the final answer is" not in reasoning
    assert reasoning.strip() == "Step 1: Calculate 5 * 6 = 30. Step 2: Add 12 to get 42."

    # Validate that validate_answer_free_prefix raises on contaminated tokens
    class MockTokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            if token_ids == [1, 2, 3]:
                return "Pure reasoning trace without answer"
            return "Reasoning trace with \\boxed{42} leaked"

    mock_tok = MockTokenizer()
    # Safe prefix passes cleanly
    validate_answer_free_prefix([1, 2, 3], mock_tok)

    # Leaked prefix raises ValueError
    import pytest
    with pytest.raises(ValueError, match="Answer leakage detected"):
        validate_answer_free_prefix([1, 2, 99], mock_tok)


def test_baseline_analysis_and_metadata():
    from src.probing.analysis import analyze_baselines

    per_example_meta = [
        {"id": "1", "label": 1, "reasoning_token_count": 50},
        {"id": "2", "label": 1, "reasoning_token_count": 60},
        {"id": "3", "label": 0, "reasoning_token_count": 40},
        {"id": "4", "label": 1, "reasoning_token_count": 70},
    ]
    probe_results = {
        10: {"auroc": 0.55},
        50: {"auroc": 0.65},
        90: {"auroc": 0.80},
    }
    record = analyze_baselines(per_example_meta, probe_results)
    assert record["n_examples"] == 4
    assert record["correct"] == 3
    assert record["incorrect"] == 1
    assert record["prevalence"] == 0.75
    assert record["majority_baseline_accuracy"] == 0.75
    assert record["early_checkpoint_auroc"] == 0.55
    assert record["late_checkpoint_auroc"] == 0.80
    assert abs(record["auroc_gain_early_to_late"] - 0.25) < 1e-6
    assert record["has_length_metadata"] is True
