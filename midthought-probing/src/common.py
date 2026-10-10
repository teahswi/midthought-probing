import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = APP_ROOT / 'config' / 'experiment.json'


def load_config(path: str | None = None) -> dict:
    config_path = Path(path) if path else CONFIG_PATH
    with open(config_path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_prompt(problem: str) -> str:
    return (
        "Solve the following math word problem carefully. Think step by step, but keep the final answer "
        "in a single, clearly identifiable box using the format \\boxed{...}.\n\n"
        f"Problem: {problem}\n\nAnswer:"
    )


def parse_numeric_answer(text: str):
    if text is None:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace('$', '').replace(',', '')
    cleaned = cleaned.replace('\\boxed{', '').replace('}', '')
    cleaned = cleaned.replace('\\displaystyle', '').strip()

    fraction_pattern = r'[-+]?\d*\.?\d+\s*/\s*[-+]?\d*\.?\d+'
    fraction_match = re.search(fraction_pattern, cleaned, flags=re.IGNORECASE)
    if fraction_match:
        fraction = fraction_match.group(0)
        nums = fraction.split('/')
        if len(nums) == 2:
            try:
                return float(nums[0].strip()) / float(nums[1].strip())
            except ValueError:
                pass

    for pattern in [
        r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?',
    ]:
        matches = re.findall(pattern, cleaned, flags=re.IGNORECASE)
        if matches:
            candidate = matches[0]
            try:
                return float(candidate)
            except ValueError:
                pass
    # Accept a final number in parentheses if no match found.
    for candidate in re.findall(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?', cleaned):
        try:
            return float(candidate)
        except ValueError:
            continue
    return None


def numeric_correctness(reference_answer: str, generated_answer: str):
    ref = parse_numeric_answer(str(reference_answer))
    gen = parse_numeric_answer(str(generated_answer))
    if ref is None or gen is None:
        return None
    if math.isnan(ref) or math.isnan(gen):
        return None
    return math.isclose(ref, gen, rel_tol=1e-6, abs_tol=1e-6)


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def to_pandas_float(value):
    try:
        return float(value)
    except Exception:
        return float('nan')


def find_reasoning_boundary(response_text: str):
    if '\\boxed{' in response_text:
        idx = response_text.rfind('\\boxed{')
        return idx
    match = re.search(r'final answer\s*[:=-]?\s*', response_text, flags=re.IGNORECASE)
    if match:
        return match.start()
    last_newline = response_text.rfind('\n')
    if last_newline > 0:
        return last_newline
    return max(0, len(response_text) - 30)


def select_snapshot_positions(total_tokens: int, percentages: list[int]) -> list[int]:
    if total_tokens <= 0:
        return []
    positions = []
    for pct in percentages:
        pct = max(0, min(100, int(pct)))
        positions.append(max(0, min(total_tokens - 1, int(round(total_tokens * (pct / 100.0))))))
    return positions


def make_manifest_record(sample_id: str, question: str, answer: str, split: str):
    return {
        'sample_id': sample_id,
        'question': question,
        'reference_answer': answer,
        'split': split,
    }
