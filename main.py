"""Config-driven CLI for the latent correctness probe workflow."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_config(path: str) -> dict:
    config_path = Path(path)
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML configs require PyYAML. Install requirements.txt.") from exc
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return json.loads(config_path.read_text(encoding="utf-8"))


def run_legacy(
    script: str,
    config_input: dict | str | Path,
    dry_run: bool = False,
    extra_args: list[str] | None = None,
) -> int:
    if isinstance(config_input, (str, Path)):
        config_path = str(Path(config_input).resolve())
    elif isinstance(config_input, dict) and "_path" in config_input:
        config_path = str(Path(config_input["_path"]).resolve())
    else:
        config_path = str(ROOT / "config" / "experiment.json")

    command = [sys.executable, str(ROOT / "src" / script), "--config", config_path]
    if extra_args:
        command.extend(extra_args)
    if dry_run:
        print("DRY RUN:", " ".join(command))
        return 0
    return subprocess.call(command)


def main() -> int:
    parser = argparse.ArgumentParser(description="Latent Correctness Probe CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "make-splits",
        "generation",
        "hidden-state-extraction",
        "probe-training",
        "baseline-evaluation",
        "intervention-run",
        "plotting",
    ):
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--split", default="validation")
        sub.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    config_path = args.config
    if args.command == "make-splits":
        return run_legacy("load_dataset.py", config_path, args.dry_run)
    if args.command == "generation":
        extra = ["--split", args.split] if args.split else None
        return run_legacy("generate_traces.py", config_path, args.dry_run, extra_args=extra)
    if args.command == "hidden-state-extraction":
        return run_legacy("extract_hidden_states.py", config_path, args.dry_run)
    if args.command == "probe-training":
        return run_legacy("train_probe.py", config_path, args.dry_run)
    if args.command == "baseline-evaluation":
        return run_legacy("evaluate.py", config_path, args.dry_run)
    if args.command == "plotting":
        return run_legacy("plot_results.py", config_path, args.dry_run)
    print("intervention-run is reserved for a future controlled intervention module.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
