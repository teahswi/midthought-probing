import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import APP_ROOT, ensure_dir, load_config


def read_csv_rows(path: str | Path):
    with open(path, 'r', encoding='utf-8', newline='') as fh:
        return list(csv.DictReader(fh))


def generate_placeholder(ax, message: str):
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12, transform=ax.transAxes)
    ax.set_axis_off()


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate the required PNG figures from saved metrics.')
    parser.add_argument('--config', default=str(APP_ROOT / 'config' / 'experiment.json'))
    parser.add_argument('--metrics-csv', default=None)
    parser.add_argument('--predictions-csv', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--dpi', type=int, default=160)
    args = parser.parse_args()

    config = load_config(args.config)
    figure_dir = Path(args.output_dir) if args.output_dir else APP_ROOT / config.get('figure_dir', 'results/figures')
    ensure_dir(figure_dir)

    metrics_path = Path(args.metrics_csv) if args.metrics_csv else APP_ROOT / config.get('output_dir', 'results') / 'metrics_by_position.csv'
    pred_path = Path(args.predictions_csv) if args.predictions_csv else APP_ROOT / config.get('output_dir', 'results') / 'predictions.csv'

    metrics_rows = read_csv_rows(metrics_path) if metrics_path.exists() else []
    pred_rows = read_csv_rows(pred_path) if pred_path.exists() else []

    fig, ax = plt.subplots(figsize=(7, 5), dpi=args.dpi)
    if metrics_rows:
        progresses = [float(row.get('progress', 0.0)) for row in metrics_rows]
        auroc = [float(row.get('auroc', 0.0)) for row in metrics_rows]
        ax.plot(progresses, auroc, marker='o', linewidth=2, label='Probe AUROC')
        ax.axhline(0.5, color='gray', linestyle='--', label='Chance baseline')
        ax.set_xlabel('Reasoning progress (%)')
        ax.set_ylabel('Held-out AUROC')
        ax.set_title('AUROC vs reasoning progress')
        ax.legend()
    else:
        generate_placeholder(ax, 'No AUROC data available')
    fig.tight_layout(); fig.savefig(figure_dir / 'auroc_vs_reasoning_progress.png', dpi=args.dpi); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=args.dpi)
    if metrics_rows:
        progresses = [float(row.get('progress', 0.0)) for row in metrics_rows]
        bal = [float(row.get('balanced_accuracy', 0.0)) for row in metrics_rows]
        maj = [float(row.get('majority_acc', 0.5)) for row in metrics_rows]
        ax.plot(progresses, bal, marker='o', linewidth=2, label='Probe')
        ax.plot(progresses, maj, marker='s', linestyle='--', label='Majority baseline')
        ax.set_xlabel('Reasoning progress (%)')
        ax.set_ylabel('Balanced accuracy')
        ax.set_title('Balanced accuracy vs reasoning progress')
        ax.legend()
    else:
        generate_placeholder(ax, 'No balanced accuracy data available')
    fig.tight_layout(); fig.savefig(figure_dir / 'balanced_accuracy_vs_progress.png', dpi=args.dpi); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=args.dpi)
    if pred_rows:
        labels = [int(float(row.get('true_label', 0))) for row in pred_rows]
        counts = {0: sum(1 for v in labels if v == 0), 1: sum(1 for v in labels if v == 1)}
        ax.bar(['Incorrect', 'Correct'], [counts.get(0, 0), counts.get(1, 0)], color=['#d95f02', '#1b9e77'])
        ax.set_title('Class distribution of generated answers')
        ax.set_ylabel('Count')
    else:
        generate_placeholder(ax, 'No class distribution data available')
    fig.tight_layout(); fig.savefig(figure_dir / 'class_distribution.png', dpi=args.dpi); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=args.dpi)
    if metrics_rows:
        progresses = [float(row.get('progress', 0.0)) for row in metrics_rows]
        auroc = [float(row.get('auroc', 0.0)) for row in metrics_rows]
        balanced = [float(row.get('balanced_accuracy', 0.0)) for row in metrics_rows]
        f1 = [float(row.get('f1', 0.0)) for row in metrics_rows]
        ax.plot(progresses, auroc, marker='o', label='AUROC')
        ax.plot(progresses, balanced, marker='s', label='Balanced accuracy')
        ax.plot(progresses, f1, marker='^', label='F1')
        ax.set_xlabel('Reasoning progress (%)')
        ax.set_ylabel('Metric value')
        ax.set_title('Probe metrics across reasoning positions')
        ax.legend()
    else:
        generate_placeholder(ax, 'No metric comparison data available')
    fig.tight_layout(); fig.savefig(figure_dir / 'probe_metrics_comparison.png', dpi=args.dpi); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=args.dpi)
    if metrics_rows:
        progresses = [float(row.get('progress', 0.0)) for row in metrics_rows]
        auroc = [float(row.get('auroc', 0.0)) for row in metrics_rows]
        ax.plot(progresses, auroc, marker='o', label='Probe AUROC', color='tab:blue')
        ax.axhline(0.5, linestyle='--', color='black', label='Chance')
        ax.set_xlabel('Reasoning progress (%)')
        ax.set_ylabel('AUROC')
        ax.set_title('Layer ablation placeholder')
        ax.legend()
    else:
        generate_placeholder(ax, 'No layer-ablation data available')
    fig.tight_layout(); fig.savefig(figure_dir / 'layer_ablation.png', dpi=args.dpi); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=args.dpi)
    if pred_rows:
        probs = [float(row.get('predicted_probability', 0.0)) for row in pred_rows]
        labels = [int(float(row.get('true_label', 0))) for row in pred_rows]
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        x_vals = []
        y_vals = []
        for left, right in zip(bins[:-1], bins[1:]):
            inside = [p for p in probs if left <= p < right or (right == 1.0 and p == 1.0)]
            if not inside:
                continue
            x_vals.append(sum(inside) / len(inside))
            y_vals.append(sum(labels[i] for i, p in enumerate(probs) if (left <= p < right or (right == 1.0 and p == 1.0))) / len(inside))
        ax.plot(x_vals, y_vals, marker='o', linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Observed frequency of correct answers')
        ax.set_title('Reliability diagram')
        ax.legend()
    else:
        generate_placeholder(ax, 'No calibration data available')
    fig.tight_layout(); fig.savefig(figure_dir / 'reliability_diagram.png', dpi=args.dpi); plt.close(fig)

    print(f'Figures written to {figure_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
