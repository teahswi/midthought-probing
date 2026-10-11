"""Prefix hidden-state extraction and token-position safeguards."""

import math

from .config import CHECKPOINT_STEPS


def snapshot_positions(total_tokens: int, checkpoints=CHECKPOINT_STEPS) -> list[int]:
    if total_tokens <= 0:
        return []
    return [
        max(0, min(total_tokens - 1, int(round(total_tokens * step / 100))))
        for step in checkpoints
    ]


def validate_answer_free_prefix(generated_token_ids, tokenizer) -> None:
    """Ensure that prefix tokens do not contain final-answer markers.

    Raises ValueError if final answer markers are detected in the token sequence.
    """
    if not generated_token_ids or tokenizer is None:
        return
    decoded = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    if r"\boxed{" in decoded or "####" in decoded:
        raise ValueError(
            "Answer leakage detected: token sequence contains final-answer text. "
            "The caller must exclude final-answer tokens before feature extraction."
        )


def extract_prefix_states(
    model,
    tokenizer,
    prompt: str,
    generated_token_ids,
    layer_index: int = -1,
    checkpoints=CHECKPOINT_STEPS,
    check_leakage: bool = True,
):
    """Return one vector per checkpoint; final-answer text must be excluded by the caller."""
    import torch

    if check_leakage and tokenizer is not None:
        validate_answer_free_prefix(generated_token_ids, tokenizer)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([prompt_ids + list(generated_token_ids)], device=model.device)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_index][0]
        # Release all other transformer layer activations from memory immediately
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    prompt_length = len(prompt_ids)
    snapshots = {}
    for step, relative_position in zip(
        checkpoints, snapshot_positions(len(generated_token_ids), checkpoints)
    ):
        absolute_position = min(prompt_length + relative_position, hidden.shape[0] - 1)
        vector = hidden[absolute_position].detach().cpu().float().numpy()
        if not all(math.isfinite(float(value)) for value in vector):
            raise ValueError(f"Non-finite hidden state at checkpoint {step}.")
        snapshots[step] = vector
    return snapshots
