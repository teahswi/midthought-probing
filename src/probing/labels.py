"""Leakage-safe correctness labels."""

from .data import answers_match


def correctness_label(reference_answer: str, generated_answer: str) -> int | None:
    matched = answers_match(reference_answer, generated_answer)
    return None if matched is None else int(matched)
