# Mid-Thought Probing: Latent Correctness Probe

This repository evaluates whether internal hidden-state signals in reasoning models predict answer correctness early enough to support inference-time analysis or control.

The core method is the **Latent Correctness Probe**: a lightweight classifier trained on intermediate transformer hidden states to predict whether a generated mathematical answer will eventually be correct. The probe is treated as a decodable signal, not as evidence that a model consciously knows its answer is wrong.

## Repository layout

```text
src/
  probing/
    analysis.py       # Baselines and result metadata audits
    collection.py     # Generation + prefix-state collection
    config.py         # Seeds and checkpoint constants
    data.py           # Dataset loading and numeric answer parsing
    features.py       # Token-position and hidden-state extraction
    generation.py     # Prompting and greedy decoding
    labels.py         # Gold-vs-generated correctness
    probes.py         # Logistic-regression probes
  plotting/
    metrics.py        # Matplotlib diagnostics
main.py              # Config-driven CLI
configs/              # Versioned YAML experiment configurations
scripts/              # Windows launcher and dataset utilities
data/                 # Local manifests and intermediate artifacts
runs/                 # Cached JSONL generations
results/              # Metrics, reports, and metadata
figs/                 # Published PNG figures
tests/                # Unit tests
```

## Important model constraint

Ollama completion APIs do not expose the per-token hidden states required by the latent-probe pipeline. Ollama models can be used for a separate generation comparison, but the full probe workflow requires the equivalent model checkpoint loaded with Hugging Face Transformers.

An Ollama model directory is also not a Transformers checkpoint. For a local Transformers model, `model.path` must point to a directory containing the tokenizer and model files.

## Windows setup

```powershell
cd midthought-probing
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

For a local checkpoint:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_local_windows.ps1 `
  -ModelPath 'C:\models\qwen3-8b'
```

Four-bit loading is only attempted on CUDA and requires a working `bitsandbytes` installation. CPU runs use float32.

## Config-driven CLI

Dry runs validate command wiring without loading a model:

```powershell
$env:PYTHONPATH = "src"
python .\main.py make-splits --config .\configs\qwen_math_smoke.yaml --dry-run
python .\main.py generation --config .\configs\qwen_math_smoke.yaml --dry-run
python .\main.py hidden-state-extraction --config .\configs\qwen_math_smoke.yaml --dry-run
```

The legacy scripts remain available directly:

```powershell
python .\src\check_environment.py --model-name "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
python .\src\run_pipeline.py --smoke-test --sample-limit 1
python .\src\run_pipeline.py --sample-limit 8 --seed 42
```

## Typical workflow

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from probing.data import load_math_split
from probing.collection import build_probe_data
from probing.probes import run_all_probes

examples = load_math_split(split="train", require_numeric=True, max_items=1500)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", device_map="auto")

features_by_t, labels_by_t, metadata = build_probe_data(
    model=model,
    tokenizer=tokenizer,
    math_data=examples,
    max_items=1500,
)
metrics = run_all_probes(features_by_t, labels_by_t, seed=42)
```

The collection code records actual token IDs and selects hidden states at token positions corresponding to reasoning progress. It never uses verbal confidence claims as features.

## Plot cached results

```python
from plotting import plot_probe_results

plot_probe_results(
    metrics,
    title="Latent Correctness Probe",
    outpath="figs/probe_curve.png",
)
```

The original plotting script can regenerate the required experiment figures:

```powershell
python .\src\plot_results.py `
  --metrics-csv .\results\metrics_by_position.csv `
  --predictions-csv .\results\predictions.csv `
  --output-dir .\figs `
  --dpi 160
```

## Reproducibility and scientific safeguards

- Use a fixed seed and preserve the exact split manifest.
- Keep training, validation, and held-out test questions disjoint.
- Do not tune thresholds, steering coefficients, or prompt policies on the final test split.
- Report unparseable answers separately instead of silently converting them to incorrect.
- Compare against majority/prevalence and metadata baselines.
- Treat results without problem IDs as **unvalidated**, not as evidence of generalization.
- A high AUROC does not establish causality, model consciousness, or a reliable intervention policy.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover numeric parsing, correctness labels, token checkpoint selection, and synthetic plot generation. Synthetic fixtures are never mixed with real research outputs.

## Outputs

Generated artifacts belong in:

- `runs/`: raw JSONL generation records
- `results/`: metrics, predictions, summaries, and metadata audits
- `figs/`: publication-quality PNG figures

Downloaded model weights, virtual environments, API keys, large hidden-state arrays, and generated traces should not be committed.
