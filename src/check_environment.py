import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from common import APP_ROOT, load_config


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def run_command(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        stdout = (completed.stdout or '').strip()
        stderr = (completed.stderr or '').strip()
        if completed.returncode != 0 and stderr:
            return f"{stderr}"
        return stdout
    except FileNotFoundError:
        return ''


def main() -> int:
    parser = argparse.ArgumentParser(description='Inspect the local environment and model availability.')
    parser.add_argument('--model-name', default=None)
    parser.add_argument('--config', default=str(APP_ROOT / 'config' / 'experiment.json'))
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = args.model_name or config.get('model_name')

    print('Environment check')
    print('Working directory:', APP_ROOT)
    print('Python:', sys.version.replace('\n', ' '))
    try:
        import torch
        print('PyTorch:', torch.__version__)
        print('CUDA available:', torch.cuda.is_available())
        print('CUDA devices:', torch.cuda.device_count())
        print('Torch CUDA version:', torch.version.cuda)
    except Exception as exc:
        print('PyTorch import failed:', exc)

    for lib_name in ['transformers', 'datasets', 'sklearn', 'pandas', 'numpy', 'matplotlib']:
        try:
            __import__(lib_name)
            print(f'{lib_name}: OK')
        except Exception as exc:
            print(f'{lib_name}: MISSING ({exc})')

    print('Ollama installed:', command_exists('ollama'))
    if command_exists('ollama'):
        models = run_command(['ollama', 'list'])
        print('Ollama models:')
        if models:
            print(models)
        else:
            print('No Ollama models are currently installed.')

    print('Default model to probe:', model_name)
    print('Local HF cache:', str(Path.home() / '.cache' / 'huggingface'))
    print('')
    print('Notes:')
    print('- Ollama does not expose intermediate hidden states needed for the main experiment.')
    print('- The default model is DeepSeek-R1-Distill-Qwen-1.5B if the environment can support it.')
    print('- If not, use the lighter debug model in config/experiment.json for pipeline validation.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
