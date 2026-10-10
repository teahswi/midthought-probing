# Mid-Thought Probing

This repository evaluates whether hidden-state signals from reasoning models can predict mathematical answer correctness before generation is complete.

The project implements a **Latent Correctness Probe**: a lightweight classifier trained on intermediate transformer representations to predict whether a generated answer will eventually be correct. The probe measures linear decodability; it does not claim that a model consciously knows whether it is correct.

## Project

The implementation is in [midthought-probing/](./midthought-probing/).

Read the complete project documentation in [midthought-probing/README.md](./midthought-probing/README.md), including:

- Windows setup and virtual-environment commands
- Config-driven experiment workflows
- Local Hugging Face model requirements
- Why Ollama cannot provide the hidden states required by the full probe
- Dataset, split, leakage, and evaluation safeguards
- Probe training and plotting examples
- Testing and reproducibility guidance

## Quick start on Windows

```powershell
cd midthought-probing
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python main.py generation --config configs\qwen_math_smoke.yaml --dry-run
```

The repository separates reusable code into:

- `src/probing/` for data, prompting, generation, hidden-state extraction, labels, and probes
- `src/plotting/` for metrics and diagnostic figures
- `configs/` for reproducible experiment settings

Real inference requires a working Transformers checkpoint and compatible PyTorch installation. Ollama models alone do not expose the per-token hidden states needed for the latent-probe pipeline.
