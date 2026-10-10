"""Small, reusable plotting functions with honest missing-result handling."""

from pathlib import Path


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_probe_results(metrics, title: str, outpath: str | Path):
    plt = _plt()
    path = Path(outpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = sorted(metrics)
    values = [metrics[step].get("auroc") for step in steps]
    valid = [(step, value) for step, value in zip(steps, values) if value is not None]
    fig, ax = plt.subplots(figsize=(7, 5))
    if valid:
        ax.plot([item[0] for item in valid], [item[1] for item in valid], marker="o", label="Probe AUROC")
        ax.axhline(0.5, linestyle="--", color="gray", label="Chance")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No valid AUROC results", ha="center", va="center")
    ax.set(xlabel="Reasoning progress (%)", ylabel="Held-out AUROC", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_baseline_comparison(records, title: str, outpath: str | Path):
    plt = _plt()
    path = Path(outpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = list(records)
    values = [records[label] for label in labels]
    ax.bar(labels, values)
    ax.set(title=title, ylabel="Metric value")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
