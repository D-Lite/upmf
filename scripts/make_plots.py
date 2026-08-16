#!/usr/bin/env python3
"""Generate dissertation figures and an evidence-based RESULTS.md."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STRATEGIES = ["upmf", "fedavg_sync", "async_vc"]
LABELS = {"upmf": "UPMF", "fedavg_sync": "FedAvg", "async_vc": "Async-VC"}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def accuracy_curves(summary: pd.DataFrame, results_dir: Path, plots_dir: Path) -> None:
    """Plot seed-aggregated high/high accuracy trajectories."""
    selected = summary[
        (summary["systems_severity"] == "high")
        & (summary["stats_severity"] == "high")
    ]
    frames = []
    for row in selected.itertuples():
        frame = pd.read_csv(results_dir / f"{row.run_id}.csv")
        frame["strategy"] = row.strategy
        frame["seed"] = row.seed
        frames.append(frame.dropna(subset=["accuracy"]))
    data = pd.concat(frames, ignore_index=True)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for strategy in STRATEGIES:
        grouped = data[data["strategy"] == strategy].groupby("round")["accuracy"]
        mean = grouped.mean()
        std = grouped.std().fillna(0.0)
        axis.plot(mean.index, mean.values, marker="o", label=LABELS[strategy])
        axis.fill_between(mean.index, mean - std, mean + std, alpha=0.15)
    axis.set(
        xlabel="Round",
        ylabel="Test accuracy",
        title="Accuracy under high systems and statistical severity",
    )
    axis.legend()
    axis.grid(alpha=0.25)
    _save(fig, plots_dir / "accuracy_vs_round.png")


def advantage_grid(summary: pd.DataFrame, plots_dir: Path) -> pd.DataFrame:
    """Plot UPMF mean accuracy advantage over the stronger mean baseline."""
    means = summary.groupby(
        ["systems_severity", "stats_severity", "strategy"]
    )["final_accuracy"].mean()
    records = []
    for systems in ["low", "high"]:
        for stats in ["low", "high"]:
            upmf = means.loc[(systems, stats, "upmf")]
            baseline = max(
                means.loc[(systems, stats, "fedavg_sync")],
                means.loc[(systems, stats, "async_vc")],
            )
            records.append(
                {"systems": systems, "stats": stats, "advantage": upmf - baseline}
            )
    frame = pd.DataFrame(records)
    grid = frame.pivot(index="systems", columns="stats", values="advantage").loc[
        ["low", "high"], ["low", "high"]
    ]
    fig, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(grid.values, cmap="coolwarm", aspect="auto")
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{grid.iloc[row, column] * 100:+.2f} pp",
                ha="center",
                va="center",
            )
    axis.set_xticks([0, 1], ["Low", "High"])
    axis.set_yticks([0, 1], ["Low", "High"])
    axis.set(
        xlabel="Statistical severity",
        ylabel="Systems severity",
        title="UPMF advantage over better baseline",
    )
    fig.colorbar(image, ax=axis, label="Accuracy difference")
    _save(fig, plots_dir / "upmf_advantage_grid.png")
    return frame


def comparison_bars(summary: pd.DataFrame, plots_dir: Path) -> None:
    """Plot wasted compute at high systems severity and overall communication."""
    high = summary[summary["systems_severity"] == "high"]
    wasted = high.groupby("strategy")["wasted_compute_fraction"].agg(["mean", "std"])
    communication = summary.groupby("strategy")["communication_bytes"].agg(
        ["mean", "std"]
    )
    for values, ylabel, title, filename, scale in [
        (
            wasted,
            "Wasted compute fraction",
            "Wasted compute at high systems severity",
            "wasted_compute.png",
            1.0,
        ),
        (
            communication,
            "Communication (MiB)",
            "Mean communication cost",
            "communication_cost.png",
            1024**2,
        ),
    ]:
        ordered = values.loc[STRATEGIES]
        fig, axis = plt.subplots(figsize=(6, 4.5))
        axis.bar(
            [LABELS[strategy] for strategy in STRATEGIES],
            ordered["mean"] / scale,
            yerr=ordered["std"].fillna(0.0) / scale,
            capsize=4,
        )
        axis.set(ylabel=ylabel, title=title)
        axis.grid(axis="y", alpha=0.25)
        _save(fig, plots_dir / filename)


def write_results(
    summary: pd.DataFrame, advantages: pd.DataFrame, output: Path
) -> None:
    """Write a plain-language interaction conclusion derived from persisted data."""
    high_high = float(
        advantages[
            (advantages["systems"] == "high") & (advantages["stats"] == "high")
        ]["advantage"].iloc[0]
    )
    maximum = float(advantages["advantage"].max())
    if high_high > 0 and np.isclose(high_high, maximum):
        conclusion = (
            "The observed results support the interaction claim: UPMF's largest "
            "mean advantage occurs in the high-systems/high-statistical condition."
        )
    elif high_high > 0:
        conclusion = (
            "The results are mixed: UPMF is ahead in the high/high condition, but "
            "its largest advantage occurs in another severity cell."
        )
    else:
        conclusion = (
            "The observed results do not support the interaction claim: UPMF does "
            "not outperform the stronger baseline in the high/high condition."
        )
    lines = [
        "# Experimental Results",
        "",
        conclusion,
        "",
        f"The high/high mean advantage is {high_high * 100:+.2f} percentage points.",
        "",
        "## Limitations",
        "",
        "- Each cell uses two random seeds, so uncertainty estimates are preliminary.",
        "- Systems-severity presets are plausible simulations, not fitted infrastructure measurements.",
        "- Results apply to MNIST, the configured small CNN, and this laptop-scale simulator.",
        "",
        f"All {len(summary)} run summaries were read from `results/summary.csv`.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Load completed results and generate all required analytical artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    results_dir = Path(args.results)
    summary = pd.read_csv(results_dir / "summary.csv")
    if len(summary) != 24:
        raise ValueError(f"Expected 24 completed runs, found {len(summary)}")
    plots_dir = results_dir / "plots"
    accuracy_curves(summary, results_dir, plots_dir)
    advantages = advantage_grid(summary, plots_dir)
    comparison_bars(summary, plots_dir)
    write_results(summary, advantages, Path("RESULTS.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
