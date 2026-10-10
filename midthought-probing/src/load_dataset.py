import argparse
import csv
import random
from pathlib import Path

from datasets import load_dataset

from common import APP_ROOT, ensure_dir, load_config, make_manifest_record, seed_everything


def write_csv(path: str | Path, fieldnames: list[str], rows: list[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(config: dict, limit: int | None = None, seed: int = 42):
    seed_everything(seed)
    dataset = load_dataset(config['dataset_name'], 'main')
    train_ds = dataset['train']
    test_ds = dataset['test']

    rows = []
    for split_name, ds in [('train', train_ds), ('test', test_ds)]:
        for i, example in enumerate(ds):
            if limit is not None and len(rows) >= limit:
                break
            rows.append(make_manifest_record(
                sample_id=f'{split_name}-{i}',
                question=example['question'],
                answer=example['answer'],
                split=split_name,
            ))
        if limit is not None and len(rows) >= limit:
            break

    if len(rows) == 0:
        raise ValueError('No rows were loaded from the GSM8K dataset.')

    train_rows = [row for row in rows if row['split'] == 'train']
    test_rows = [row for row in rows if row['split'] == 'test']

    if len(train_rows) > 1:
        random.Random(seed).shuffle(train_rows)
        train_split = train_rows[:max(1, int(len(train_rows) * 0.8))]
        val_split = train_rows[max(1, int(len(train_rows) * 0.8)):]
        final_rows = [
            {**row, 'split': 'train'} for row in train_split
        ] + [
            {**row, 'split': 'val'} for row in val_split
        ] + [
            {**row, 'split': 'test'} for row in test_rows
        ]
    else:
        final_rows = rows

    fieldnames = ['sample_id', 'question', 'reference_answer', 'split']
    manifest_dir = ensure_dir(APP_ROOT / config.get('manifest_dir', 'data/manifests'))
    manifest_path = manifest_dir / 'dataset_manifest.csv'
    write_csv(manifest_path, fieldnames, [{
        'sample_id': row['sample_id'],
        'question': row['question'],
        'reference_answer': row['reference_answer'],
        'split': row['split'],
    } for row in final_rows])
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description='Load GSM8K and create the dataset manifest.')
    parser.add_argument('--config', default=str(APP_ROOT / 'config' / 'experiment.json'))
    parser.add_argument('--sample-limit', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = build_manifest(config, limit=args.sample_limit, seed=args.seed)
    print(f'Dataset manifest saved to {manifest_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
