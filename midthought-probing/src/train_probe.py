import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import APP_ROOT, ensure_dir, load_config, numeric_correctness, seed_everything


def read_csv_rows(path: str | Path):
    with open(path, 'r', encoding='utf-8', newline='') as fh:
        return list(csv.DictReader(fh))


def write_csv(path: str | Path, fieldnames: list[str], rows: list[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'brier_score': brier_score_loss(y_true, y_prob),
        'auroc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan'),
    }
    return metrics


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

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced', random_state=args.seed)
        clf.fit(X_train_scaled, y_train)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]

        metrics = compute_metrics(y_test, y_prob)
        metrics['progress'] = float(progress)
        metrics['n_train'] = len(train_rows)
        metrics['n_test'] = len(test_rows)
        metrics['positive_rate'] = float(y_test.mean())
        metrics['majority_acc'] = max(float(y_test.mean()), 1.0 - float(y_test.mean()))
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
