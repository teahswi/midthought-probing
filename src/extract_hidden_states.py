import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import APP_ROOT, ensure_dir, find_reasoning_boundary, load_config


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


def load_model(model_name: str, device: str = 'cpu'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def extract_snapshot_features(question: str, generated_response: str, tokenizer, model, device: str, layer_index: int = -1, percentages=None):
    if percentages is None:
        percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    prompt = "Solve the following math word problem carefully. Think step by step, but keep the final answer in a single, clearly identifiable box using the format \\boxed{...}.\n\nProblem: " + question + "\n\nAnswer:"
    full_text = prompt + generated_response
    inputs = tokenizer(full_text, return_tensors='pt', add_special_tokens=False).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    all_hidden = outputs.hidden_states
    layer_id = len(all_hidden) + layer_index if layer_index < 0 else layer_index
    output_states = all_hidden[layer_id]
    hidden_matrix = output_states[0].detach().cpu().numpy()
    del all_hidden, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    prompt_input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
    prompt_length = len(prompt_input_ids)

    # Strictly isolate reasoning tokens prior to the final answer boundary
    boundary = find_reasoning_boundary(generated_response)
    reasoning_text = generated_response[:boundary].rstrip() if boundary > 0 else generated_response
    reasoning_tokens = tokenizer(reasoning_text, add_special_tokens=False, return_attention_mask=False)['input_ids']
    reasoning_positions = [int(round(len(reasoning_tokens) * ratio / 100.0)) for ratio in percentages]
    allowed = []
    for progress, reasoning_index in zip(percentages, reasoning_positions):
        if len(reasoning_tokens) <= 0:
            continue
        reasoning_index = max(0, min(len(reasoning_tokens) - 1, reasoning_index))
        token_position = prompt_length + reasoning_index
        if token_position >= hidden_matrix.shape[0]:
            token_position = hidden_matrix.shape[0] - 1
        feature_vector = hidden_matrix[token_position].astype(float)
        if not (feature_vector == feature_vector).all():
            raise ValueError('Feature vector contains NaN values for a hidden-state snapshot.')
        allowed.append({
            'progress': progress,
            'reasoning_index': reasoning_index,
            'token_position': int(token_position),
            'feature_vector': feature_vector.tolist(),
        })
    return allowed


def main() -> int:
    parser = argparse.ArgumentParser(description='Extract hidden-state snapshots at fixed reasoning-progress checkpoints.')
    parser.add_argument('--config', default=str(APP_ROOT / 'config' / 'experiment.json'))
    parser.add_argument('--trace-csv', default=None)
    parser.add_argument('--model-name', default=None)
    parser.add_argument('--layer-index', type=int, default=-1)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    trace_path = Path(args.trace_csv) if args.trace_csv else APP_ROOT / config.get('trace_dir', 'data/traces') / 'trace_records.csv'
    output_dir = Path(args.output_dir) if args.output_dir else APP_ROOT / config.get('hidden_state_dir', 'data/hidden_states')
    ensure_dir(output_dir)

    records = read_csv_rows(trace_path)
    model_name = args.model_name or config.get('debug_model_name') or config.get('model_name')
    device = config.get('device', 'cpu')
    tokenizer, model = load_model(model_name, device=device)

    rows = []
    for row in records:
        snapshots = extract_snapshot_features(
            question=row['question'],
            generated_response=row['generated_response'],
            tokenizer=tokenizer,
            model=model,
            device=device,
            layer_index=args.layer_index,
            percentages=config.get('snapshot_percentages', [10, 20, 30, 40, 50, 60, 70, 80, 90]),
        )
        for snapshot in snapshots:
            rows.append({
                'sample_id': row['sample_id'],
                'split': row.get('split', 'test'),
                'progress': snapshot['progress'],
                'token_position': snapshot['token_position'],
                'reasoning_index': snapshot['reasoning_index'],
                'layer_index': args.layer_index,
                'feature_vector': json.dumps(snapshot['feature_vector']),
                'reference_answer': row.get('reference_answer', ''),
                'generated_answer': row.get('final_answer', ''),
            })

    feature_path = output_dir / 'hidden_state_snapshots.csv'
    write_csv(feature_path, [
        'sample_id', 'split', 'progress', 'token_position', 'reasoning_index', 'layer_index', 'feature_vector', 'reference_answer', 'generated_answer'
    ], rows)
    print(f'Hidden-state snapshots saved to {feature_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
