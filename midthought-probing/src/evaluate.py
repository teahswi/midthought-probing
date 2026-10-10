import argparse
import csv
from pathlib import Path

from common import APP_ROOT, ensure_dir, load_config, save_json


def read_csv_rows(path: str | Path):
    with open(path, 'r', encoding='utf-8', newline='') as fh:
        return list(csv.DictReader(fh))


def build_summary(metrics_rows, config):
    if not metrics_rows:
        return {
            'status': 'not_run',
            'message': 'No metrics were available. The experiment did not complete.',
            'best_auroc': None,
            'best_progress': None,
        }
    best = max(metrics_rows, key=lambda row: float(row.get('auroc', -1.0)))
    return {
        'status': 'completed',
        'best_auroc': float(best.get('auroc', 'nan')),
        'best_progress': float(best.get('progress', 'nan')),
        'n_positions': len(metrics_rows),
        'config': config,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Summarize experimental metrics into JSON and Markdown reports.')
    parser.add_argument('--config', default=str(APP_ROOT / 'config' / 'experiment.json'))
    parser.add_argument('--metrics-csv', default=None)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else APP_ROOT / config.get('output_dir', 'results')
    ensure_dir(output_dir)
    metrics_path = Path(args.metrics_csv) if args.metrics_csv else output_dir / 'metrics_by_position.csv'

    metrics_rows = read_csv_rows(metrics_path) if metrics_path.exists() else []
    summary = build_summary(metrics_rows, config)
    save_json(output_dir / 'summary.json', summary)
    save_json(output_dir / 'experiment_config.json', config)

    report_lines = [
        '# Mid-Thought Probing Report',
        '',
        '## Research question',
        'Can a lightweight logistic regression classifier trained on intermediate hidden states predict whether a reasoning model\'s final mathematical answer will be correct before the reasoning trace is complete?',
        '',
        '## Hardware and software',
        f"- Model default: {config.get('model_name')}",
        f"- Debug model: {config.get('debug_model_name')}",
        f"- Seed: {config.get('seed')}",
        '',
        '## Methodology',
        '- Use the GSM8K official test set as the held-out evaluation split.',
        '- Use hidden-state snapshots from fixed reasoning-progress checkpoints.',
        '- Fit a logistic regression probe on the training split and evaluate on the held-out set.',
        '- Keep the final answer text separate from the probe input.',
        '',
    ]

    if metrics_rows:
        best = max(metrics_rows, key=lambda row: float(row.get('auroc', -1.0)))
        report_lines.extend([
            '## Measured results',
            f"- Best AUROC: {best.get('auroc', 'n/a')}",
            f"- Best progress: {best.get('progress', 'n/a')}%",
            '',
        ])
    else:
        report_lines.extend([
            '## Measured results',
            '- No measured results were available because the experiment was not completed.',
            '',
        ])

    report_lines.extend([
        '## Interpretation',
        'If AUROC remains near chance (0.5), this suggests that hidden states do not provide a strong predictive signal for final answer correctness at the selected checkpoints. A rising curve may suggest that correctness-related information becomes more decodable as the reasoning trace unfolds, but it does not imply causality or consciousness-level certainty.',
        '',
        '## Limitations',
        '- Compute constraints may limit the number of examples and layers evaluated.',
        '- Hidden-state extraction is performed at discrete reasoning checkpoints rather than every generated token.',
        '- Model output quality and answer formatting may affect correctness detection.',
        '',
        '## Reproducibility',
        'Use the commands in README.md to rerun the environment check, smoke test, and full pipeline.',
    ])

    (output_dir / 'REPORT.md').write_text('\n'.join(report_lines) + '\n', encoding='utf-8')
    print(f'Summary saved to {output_dir / "summary.json"}')
    print(f'Report saved to {output_dir / "REPORT.md"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
