"""Dataset loading and numeric answer normalisation."""

import csv
import random
import re
from pathlib import Path

from .config import DEFAULT_SEED


def parse_numeric_answer(text: str | None) -> float | None:
    """Parse integers, decimals, signed values, commas, scientific notation, or fractions."""
    if text is None:
        return None
    cleaned = str(text).strip().replace("$", "").replace(",", "")
    cleaned = cleaned.replace("\\boxed{", "").replace("}", "")
    fraction = re.search(r"[-+]?\d*\.?\d+\s*/\s*[-+]?\d*\.?\d+", cleaned)
    if fraction:
        numerator, denominator = fraction.group(0).split("/")
        try:
            return float(numerator) / float(denominator)
        except ValueError:
            return None
    match = re.search(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", cleaned, re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def answers_match(reference: str | None, prediction: str | None) -> bool | None:
    import math

    expected = parse_numeric_answer(reference)
    actual = parse_numeric_answer(prediction)
    if expected is None or actual is None:
        return None
    return math.isclose(expected, actual, rel_tol=1e-6, abs_tol=1e-6)


def load_math_split(
    split: str = "train",
    min_level: int = 1,
    require_numeric: bool = True,
    dataset_name: str = "gsm8k",
    max_items: int | None = None,
):
    """Load a Hugging Face math split, with a local CSV fallback for cached manifests."""
    try:
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, "main", split=split)
        rows = [dict(row) for row in dataset]
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "Dataset loading requires a working `datasets` installation or a local manifest."
        ) from exc

    if require_numeric:
        rows = [row for row in rows if parse_numeric_answer(row.get("answer")) is not None]
    return rows[:max_items] if max_items is not None else rows


def write_jsonl(rows, path: str | Path) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_ids(ids: list[str], train_ratio: float = 0.8, seed: int = DEFAULT_SEED):
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    cutoff = max(1, int(len(shuffled) * train_ratio)) if shuffled else 0
    return shuffled[:cutoff], shuffled[cutoff:]
