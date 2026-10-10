"""End-to-end collection helpers for generation and prefix features."""

from .config import CHECKPOINT_STEPS
from .features import extract_prefix_states
from .generation import generate_greedy
from .labels import correctness_label


def build_probe_data(
    model,
    tokenizer,
    math_data,
    max_items: int | None = None,
    max_new_tokens: int = 256,
    layer_index: int = -1,
):
    features_by_t = {step: [] for step in CHECKPOINT_STEPS}
    labels_by_t = {step: [] for step in CHECKPOINT_STEPS}
    metadata = []
    selected = math_data[:max_items] if max_items is not None else math_data
    for index, example in enumerate(selected):
        generated = generate_greedy(model, tokenizer, example["question"], max_new_tokens)
        label = correctness_label(example["answer"], generated["final_answer"])
        if label is None:
            continue
        snapshots = extract_prefix_states(
            model,
            tokenizer,
            generated["prompt"],
            generated["token_ids"],
            layer_index=layer_index,
        )
        for step, vector in snapshots.items():
            features_by_t[step].append(vector)
            labels_by_t[step].append(label)
        metadata.append(
            {
                "example_id": example.get("id", str(index)),
                "label": label,
                "generated_answer": generated["final_answer"],
            }
        )
    return features_by_t, labels_by_t, metadata
