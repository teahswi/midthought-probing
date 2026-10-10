# Mid-Thought Probing for Early Prediction of Reasoning Failure

This project tests whether hidden representations from a reasoning model carry information about whether the model's eventual mathematical answer will be correct before reasoning is complete.

## Quick start

```powershell
cd "c:\Users\tejas\llm confidence probe\midthought-probing"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src\check_environment.py --model-name "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
python src\run_pipeline.py --smoke-test --sample-limit 1 --model-name "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
```

## Smallest viable configuration before downloading large model weights

- CPU-only environment: use `--model-name` to point to a smaller or locally cached model; the default configuration targets `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` but only if it can load without OOM.
- If Hugging Face weights are unavailable or the environment is too constrained, use the debug fallback model configured in `config/experiment.json` for pipeline validation.
- Ollama is only an optional baseline for output generation and does not expose the hidden states needed for the probe.

## Project structure

- `src/` contains the pipeline scripts.
- `config/experiment.json` stores default settings.
- `results/` stores metrics, predictions, summary documents, and generated figures.
- `tests/` contains automated tests.

## Reproducibility

Run the full pipeline with:

```powershell
python src\run_pipeline.py --sample-limit 8 --seed 42 --output-dir results --model-name "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
```

To regenerate plots from saved results only:

```powershell
python src\plot_results.py --metrics-csv results\metrics_by_position.csv --predictions-csv results\predictions.csv --output-dir results\figures --dpi 160
```

## Notes

- The experiment is intentionally designed to avoid evaluating on the same examples used for training.
- Correctness is determined by a robust numeric parser over the final answer text.
- Hidden states are extracted from exact token positions at fixed reasoning-progress checkpoints.
- Unit tests use synthetic fixtures and never mix synthetic data with real evaluation results.
