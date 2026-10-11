import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from common import APP_ROOT, ensure_dir, load_config, numeric_correctness, seed_everything
sys.path.insert(0, str(APP_ROOT / "src"))
from probing.probes import run_probe


def main() -> int:
    parser = argparse.ArgumentParser(description='Train a logistic-regression probe from hidden-state snapshots.')
    parser.add_argument('--config', default=str(APP_ROOT / 'config' / 'experiment.json'))
    parser.add_argument('--hidden-csv', default=None)
    parser.add_argument('--trace-csv', default=None)
    parser.add_argument('--dataset-manifest', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(args.seed)

    hidden_path = Path(args.hidden_csv) if args.hidden_csv else APP_ROOT / config.get('hidden_state_dir', 'data/hidden_states') / 'hidden_state_snapshots.csv'
    trace_path = Path(args.trace_csv) if args.trace_csv else APP_ROOT / config.get('trace_dir', 'data/traces') / 'trace_records.csv'
    manifest_path = Path(args.dataset_manifest) if args.dataset_manifest else APP_ROOT / config.get('manifest_dir', 'data/manifests') / 'dataset_manifest.csv'
    output_dir = Path(args.output_dir) if args.output_dir else APP_ROOT / config.get('output_dir', 'results')
    ensure_dir(output_dir)

    manifest = read_csv_rows(manifest_path)
    traces = read_csv_rows(trace_path)
    for row in traces:
        if 'final_answer' not in row:
            row['final_answer'] = ''
        row['correctness'] = numeric_correctness(row.get('reference_answer', ''), row.get('final_answer', ''))
    hidden_rows = read_csv_rows(hidden_path)
    for row in hidden_rows:
        row['feature_vector'] = json.loads(row['feature_vector']) if isinstance(row['feature_vector'], str) else row['feature_vector']

    merged = []
    trace_index = {row['sample_id']: row for row in traces}
    manifest_index = {row['sample_id']: row for row in manifest}
    for hidden in hidden_rows:
        sample_id = hidden['sample_id']
        trace_row = trace_index.get(sample_id, {})
        manifest_row = manifest_index.get(sample_id, {})
        merged.append({
            **hidden,
            'correctness': int(1 if trace_row.get('correctness') is True else 0 if trace_row.get('correctness') is False else 0),
            'split': manifest_row.get('split', hidden.get('split', 'test')),
        })

    progress_values = sorted({int(row['progress']) for row in merged})
    rows = []
    predictions = []

    for progress in progress_values:
        subset = [row for row in merged if int(row['progress']) == progress]
        train_rows = [row for row in subset if row['split'] == 'train']
        test_rows = [row for row in subset if row['split'] == 'test']

        if not train_rows or not test_rows:
            continue

        X_train = np.vstack([np.asarray(row['feature_vector'], dtype=float) for row in train_rows])
        y_train = np.asarray([int(row['correctness']) for row in train_rows], dtype=int)
        X_test = np.vstack([np.asarray(row['feature_vector'], dtype=float) for row in test_rows])
        y_test = np.asarray([int(row['correctness']) for row in test_rows], dtype=int)

        metrics = run_probe(X_train, y_train, X_test, y_test, seed=args.seed)
        y_prob = np.asarray(metrics['probabilities'])
        pred = (y_prob >= 0.5).astype(int)
        metrics['accuracy'] = float(np.mean(pred == y_test))
        metrics['progress'] = float(progress)
        metrics['n_train'] = len(train_rows)
        metrics['n_test'] = len(test_rows)
        metrics['positive_rate'] = float(y_test.mean())
        metrics['majority_acc'] = metrics.get('majority_acc', max(float(y_test.mean()), 1.0 - float(y_test.mean())))
        rows.append(metrics)

        for record, prob in zip(test_rows, y_prob):
            predictions.append({
                'sample_id': record['sample_id'],
                'progress': progress,
                'true_label': int(record['correctness']),
                'predicted_probability': float(np.clip(prob, 0.0, 1.0)),
                'split': 'test',
            })

    metrics_path = output_dir / 'metrics_by_position.csv'
    write_csv(metrics_path, ['progress','accuracy','balanced_accuracy','precision','recall','f1','brier_score','auroc','n_train','n_test','positive_rate','majority_acc'], [
        {
            'progress': row['progress'],
            'accuracy': row['accuracy'],
            'balanced_accuracy': row['balanced_accuracy'],
            'precision': row['precision'],
            'recall': row['recall'],
            'f1': row['f1'],
            'brier_score': row['brier_score'],
            'auroc': row['auroc'],
            'n_train': row['n_train'],
            'n_test': row['n_test'],
            'positive_rate': row['positive_rate'],
            'majority_acc': row['majority_acc'],
        } for row in rows
    ])

    pred_path = output_dir / 'predictions.csv'
    write_csv(pred_path, ['sample_id','progress','true_label','predicted_probability','split'], predictions)

    print(f'Metrics saved to {metrics_path}')
    print(f'Predictions saved to {pred_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
