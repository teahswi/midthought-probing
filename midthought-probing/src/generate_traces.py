import argparse
import csv
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import APP_ROOT, build_prompt, ensure_dir, load_config, parse_numeric_answer, seed_everything


def read_manifest(path: str | Path):
    rows = []
    with open(path, 'r', encoding='utf-8', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def write_csv(path: str | Path, fieldnames: list[str], rows: list[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_final_answer(response: str):
    match = re.search(r'\\boxed\{([^}]*)\}', response)
    if match:
        return match.group(1).strip()
    text = response.strip()
    if ':' in text:
        parts = text.rsplit(':', 1)
        if len(parts) == 2 and parts[1].strip():
            return parts[1].strip()
    return text.splitlines()[-1].strip() if text else ''


def load_model(model_name: str, device: str = 'cpu'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def generate_trace_row(model_name: str, question: str, reference_answer: str, tokenizer, model, device: str, max_new_tokens: int):
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = output[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    final_answer = extract_final_answer(response)
    return {
        'question': question,
        'prompt': prompt,
        'generated_response': response,
        'final_answer': final_answer,
        'reference_answer': reference_answer,
        'model_name': model_name,
        'answer_is_parseable': parse_numeric_answer(final_answer) is not None,
        'correctness': '',
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate reasoning traces for the GSM8K dataset.')
    parser.add_argument('--config', default=str(APP_ROOT / 'config' / 'experiment.json'))
    parser.add_argument('--split', default='test')
    parser.add_argument('--sample-limit', type=int, default=None)
    parser.add_argument('--model-name', default=None)
    parser.add_argument('--use-ollama', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(args.seed)
    manifest_path = APP_ROOT / config.get('manifest_dir', 'data/manifests') / 'dataset_manifest.csv'
    manifest = read_manifest(manifest_path)
    selected_rows = [row for row in manifest if row['split'] == args.split] if args.split in {row['split'] for row in manifest} else manifest
    if args.sample_limit is not None:
        selected_rows = selected_rows[:args.sample_limit]

    model_name = args.model_name or config.get('debug_model_name') or config.get('model_name')
    device = config.get('device', 'cpu')
    max_new_tokens = args.max_new_tokens or config.get('max_new_tokens', 256)

    tokenizer, model = load_model(model_name=model_name, device=device)

    records = []
    for row in selected_rows:
        trace = generate_trace_row(
            model_name=model_name,
            question=row['question'],
            reference_answer=row['reference_answer'],
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        trace['sample_id'] = row['sample_id']
        trace['split'] = row['split']
        records.append(trace)

    output_dir = ensure_dir(APP_ROOT / config.get('trace_dir', 'data/traces'))
    output_path = output_dir / 'trace_records.csv'
    write_csv(output_path, [
        'sample_id', 'split', 'question', 'reference_answer', 'prompt', 'generated_response',
        'final_answer', 'model_name', 'answer_is_parseable', 'correctness'
    ], [{
        'sample_id': row['sample_id'],
        'split': row['split'],
        'question': row.get('question', ''),
        'reference_answer': row.get('reference_answer', ''),
        'prompt': row.get('prompt', ''),
        'generated_response': row.get('generated_response', ''),
        'final_answer': row.get('final_answer', ''),
        'model_name': row.get('model_name', ''),
        'answer_is_parseable': str(row.get('answer_is_parseable', '')),
        'correctness': row.get('correctness', ''),
    } for row in records])
    print(f'Trace export complete: {output_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
