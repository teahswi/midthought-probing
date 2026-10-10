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
