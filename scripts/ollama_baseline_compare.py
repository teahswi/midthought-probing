"""Compare two Ollama generation baselines on real GSM8K test questions.

This is intentionally not a latent hidden-state experiment: Ollama's completion
API does not expose the per-token hidden states required for the probe.
"""

import argparse
import json
import math
import re
import time
import urllib.request
from pathlib import Path


def parse_number(text):
    cleaned = str(text or "").replace(",", "").replace("$", "")
    boxed = re.findall(r"\\boxed\{([^}]*)\}", cleaned)
    if boxed:
        cleaned = boxed[-1]
    fractions = re.findall(r"[-+]?\d+(?:\.\d+)?\s*/\s*[-+]?\d+(?:\.\d+)?", cleaned)
    if fractions:
        left, right = fractions[-1].split("/")
        try:
            return float(left) / float(right)
        except ValueError:
            return None
    values = re.findall(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not values:
        return None
    try:
        return float(values[-1])
    except ValueError:
        return None


def reference_number(answer):
    marker = "####"
    return parse_number(answer.split(marker, 1)[-1] if marker in answer else answer)


def call_ollama(model, prompt, timeout):
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def prompt_for(question):
    return (
        "Solve this GSM8K math problem. Show concise work and end with the final "
        "numeric answer after ####.\n\nQuestion:\n" + question
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gsm8k_test.jsonl")
    parser.add_argument("--models", nargs=2, default=["qwen3:1.7b", "phi3:mini"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output-dir", default="results/ollama_baseline")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    questions = []
    with open(args.data, encoding="utf-8") as handle:
        for line in handle:
            if len(questions) >= args.limit:
                break
            questions.append(json.loads(line))

    records = []
    for model in args.models:
        for index, example in enumerate(questions):
            started = time.time()
            try:
                result = call_ollama(model, prompt_for(example["question"]), args.timeout)
                response = result.get("response", "")
                prediction = parse_number(response)
                expected = reference_number(example["answer"])
                correct = (
                    prediction is not None
                    and expected is not None
                    and math.isclose(prediction, expected, rel_tol=1e-6, abs_tol=1e-6)
                )
                error = ""
            except Exception as exc:
                response = ""
                prediction = None
                correct = False
                error = f"{type(exc).__name__}: {exc}"
            records.append(
                {
                    "model": model,
                    "question_index": index,
                    "correct": int(correct),
                    "prediction": prediction,
                    "response": response,
                    "error": error,
                    "elapsed_seconds": round(time.time() - started, 3),
                }
            )
            print(f"{model}: {index + 1}/{len(questions)} correct={correct}")

    records_path = output_dir / "predictions.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summaries = []
    for model in args.models:
        model_rows = [row for row in records if row["model"] == model]
        correct = sum(row["correct"] for row in model_rows)
        summaries.append(
            {
                "model": model,
                "n_questions": len(model_rows),
                "correct": correct,
                "incorrect_or_unparseable": len(model_rows) - correct,
                "accuracy": correct / len(model_rows) if model_rows else None,
                "errors": sum(bool(row["error"]) for row in model_rows),
                "hidden_state_probe": False,
            }
        )
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["model"] for row in summaries]
    accuracies = [row["accuracy"] or 0 for row in summaries]
    counts = [row["correct"] for row in summaries]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(labels, accuracies, color=["#4472c4", "#ed7d31"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Exact numeric accuracy")
    axes[0].set_title("Ollama generation baseline")
    axes[1].bar(labels, counts, color=["#4472c4", "#ed7d31"])
    axes[1].set_ylabel("Correct answers")
    axes[1].set_title(f"Correct answers ({len(questions)} GSM8K test questions)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
    fig.suptitle("Output-only comparison; not hidden-state probe results", fontsize=12)
    fig.tight_layout()
    fig.savefig(figure_dir / "model_accuracy_comparison.png", dpi=160)
    plt.close(fig)

    print(f"Summary: {output_dir / 'summary.json'}")
    print(f"Predictions: {records_path}")
    print(f"PNG: {figure_dir / 'model_accuracy_comparison.png'}")


if __name__ == "__main__":
    main()
