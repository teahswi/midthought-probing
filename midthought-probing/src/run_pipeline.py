import argparse
from pathlib import Path

from common import APP_ROOT, load_config, seed_everything


def run_command(command: list[str]):
    import subprocess
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(f'Command failed: {command} (exit={result.returncode})')


def main() -> int:
    parser = argparse.ArgumentParser(description='Run the end-to-end mid-thought probing pipeline.')
    parser.add_argument('--config', default=str(APP_ROOT / 'config' / 'experiment.json'))
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--sample-limit', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model-name', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--debug-model-name', default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(args.seed)

    limit = args.sample_limit if args.sample_limit is not None else (config.get('smoke_limit') if args.smoke_test else config.get('sample_limit'))
    model_name = args.model_name or config.get('model_name')
    debug_model = args.debug_model_name or config.get('debug_model_name')

    if args.smoke_test:
        print('Running smoke test')
        model_name = debug_model
    print(f'Using model: {model_name}')

    run_command(['python', str(APP_ROOT / 'src' / 'load_dataset.py'), '--config', args.config, '--sample-limit', str(limit), '--seed', str(args.seed)])
    run_command(['python', str(APP_ROOT / 'src' / 'generate_traces.py'), '--config', args.config, '--split', 'test', '--sample-limit', str(limit), '--model-name', model_name, '--seed', str(args.seed)])
    run_command(['python', str(APP_ROOT / 'src' / 'extract_hidden_states.py'), '--config', args.config, '--trace-csv', str(APP_ROOT / config.get('trace_dir', 'data/traces') / 'trace_records.csv'), '--model-name', model_name, '--output-dir', str(APP_ROOT / config.get('hidden_state_dir', 'data/hidden_states'))])
    run_command(['python', str(APP_ROOT / 'src' / 'train_probe.py'), '--config', args.config, '--hidden-csv', str(APP_ROOT / config.get('hidden_state_dir', 'data/hidden_states') / 'hidden_state_snapshots.csv'), '--trace-csv', str(APP_ROOT / config.get('trace_dir', 'data/traces') / 'trace_records.csv'), '--dataset-manifest', str(APP_ROOT / config.get('manifest_dir', 'data/manifests') / 'dataset_manifest.csv'), '--seed', str(args.seed), '--output-dir', str(APP_ROOT / config.get('output_dir', 'results'))])
    run_command(['python', str(APP_ROOT / 'src' / 'evaluate.py'), '--config', args.config, '--metrics-csv', str(APP_ROOT / config.get('output_dir', 'results') / 'metrics_by_position.csv'), '--output-dir', str(APP_ROOT / config.get('output_dir', 'results'))])
    run_command(['python', str(APP_ROOT / 'src' / 'plot_results.py'), '--config', args.config, '--metrics-csv', str(APP_ROOT / config.get('output_dir', 'results') / 'metrics_by_position.csv'), '--predictions-csv', str(APP_ROOT / config.get('output_dir', 'results') / 'predictions.csv'), '--output-dir', str(APP_ROOT / config.get('figure_dir', 'results/figures'))])

    print('Pipeline complete.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
