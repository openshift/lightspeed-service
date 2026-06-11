r"""Record eval results and regenerate weekly progression plots.

Reads one or more lightspeed-eval *_summary.json files, appends a row per
metric to eval/score_history.csv, then writes two trend PNGs into
--output-dir:

  trend_pass_rate.png  — pass_rate over time, one line per metric per suite
  trend_score_mean.png — score_mean over time, one line per metric per suite

Usage::

    python eval/scripts/update_eval_trends.py \
        --suite lseval_periodic \
        --summary-json /logs/artifacts/lseval/openai/evaluation_YYYYMMDD_summary.json \
        --suite lseval_troubleshooting_scenarios \
        --summary-json /logs/artifacts/troubleshooting/scenarios/evaluation_YYYYMMDD_summary.json \
        --suite lseval_troubleshooting_mcp \
        --summary-json /logs/artifacts/troubleshooting/mcp/evaluation_YYYYMMDD_summary.json \
        --suite lseval_netobserv \
        --summary-json /logs/artifacts/netobserv/evaluation_YYYYMMDD_summary.json \
        --history-csv eval/score_history.csv \
        --output-dir /logs/artifacts \
        [--date 2026-04-30]

--suite and --summary-json are paired in order; pass the pair once per suite
that ran in this CI job (omit suites whose JSON does not exist).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

_HISTORY_COLUMNS = [
    "date",
    "suite",
    "metric",
    "total",
    "pass_rate",
    "error_rate",
    "score_mean",
]

_SUITE_LABELS: dict[str, str] = {
    "lseval_periodic": "Periodic QnA (797 questions)",
    "lseval_troubleshooting_scenarios": "Troubleshooting Scenarios",
    "lseval_troubleshooting_mcp": "Troubleshooting MCP",
    "lseval_netobserv": "NetObserv Scenarios",
}

_METRIC_LABELS: dict[str, str] = {
    "custom:answer_correctness": "Answer Correctness",
    "geval:generic_troubleshooting_experience": "Troubleshooting Experience",
    "geval:troubleshooting_continuity": "Troubleshooting Continuity",
    "deepeval:conversation_completeness": "Conversation Completeness",
    "deepeval:conversation_relevancy": "Conversation Relevancy",
    "deepeval:knowledge_retention": "Knowledge Retention",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        action="append",
        dest="suites",
        metavar="NAME",
        required=True,
        help="Suite name (repeatable, paired with --summary-json)",
    )
    parser.add_argument(
        "--summary-json",
        action="append",
        dest="jsons",
        metavar="PATH",
        required=True,
        type=Path,
        help="Path to *_summary.json (repeatable, paired with --suite)",
    )
    parser.add_argument("--history-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--date",
        default=None,
        help="ISO date (YYYY-MM-DD); defaults to timestamp in each JSON",
    )
    args = parser.parse_args()
    if len(args.suites) != len(args.jsons):
        parser.error(
            "--suite and --summary-json must be provided the same number of times"
        )
    return args


# ---------------------------------------------------------------------------
# History CSV
# ---------------------------------------------------------------------------


def _run_date(summary: dict, override: str | None) -> str:
    if override:
        return override
    ts: str = summary.get("timestamp", "")
    return ts[:10] if ts else "unknown"


def _rows_from_summary(summary: dict, suite: str, date: str) -> list[dict]:
    stats = summary.get("summary_stats", {})
    overall = stats.get("overall", {})
    total = overall.get("TOTAL", 0)
    pass_rate = round(overall.get("pass_rate", 0.0), 2)
    error_rate = round(overall.get("error_rate", 0.0), 2)

    rows = []
    for metric, mstats in stats.get("by_metric", {}).items():
        mean = (mstats.get("score_statistics") or {}).get("mean")
        rows.append(
            {
                "date": date,
                "suite": suite,
                "metric": metric,
                "total": total,
                "pass_rate": pass_rate,
                "error_rate": error_rate,
                "score_mean": round(mean, 4) if mean is not None else "",
            }
        )

    if not rows:
        rows.append(
            {
                "date": date,
                "suite": suite,
                "metric": "",
                "total": total,
                "pass_rate": pass_rate,
                "error_rate": error_rate,
                "score_mean": "",
            }
        )
    return rows


def _append_to_history(history_csv: Path, rows: list[dict]) -> None:
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not history_csv.exists() or history_csv.stat().st_size == 0
    with open(history_csv, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_HISTORY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Trend plots
# ---------------------------------------------------------------------------


def _plot_metric(
    df: pd.DataFrame, column: str, title: str, ylabel: str, output_path: Path
) -> None:
    data = df.dropna(subset=[column])
    if data.empty:
        print(f"No data for '{column}', skipping {output_path.name}")
        return

    suites = data["suite"].unique()
    fig, axes = plt.subplots(
        len(suites), 1, figsize=(12, 4 * len(suites)), squeeze=False
    )

    for ax, suite in zip(axes[:, 0], suites):
        subset = data[data["suite"] == suite]
        ax.set_title(_SUITE_LABELS.get(suite, suite), fontsize=11, fontweight="bold")
        for metric in subset["metric"].unique():
            mdf = subset[subset["metric"] == metric]
            ax.plot(
                mdf["date"],
                mdf[column],
                marker="o",
                linewidth=1.8,
                markersize=5,
                label=_METRIC_LABELS.get(metric, metric),
            )
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, 105 if "rate" in column else 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _generate_plots(history_csv: Path, output_dir: Path) -> None:
    if not history_csv.is_file():
        print(
            f"No history CSV at {history_csv} — skipping trend plots.",
            file=sys.stderr,
        )
        return
    try:
        df = pd.read_csv(history_csv, parse_dates=["date"])
    except (FileNotFoundError, PermissionError, OSError) as e:
        print(
            f"Could not read history CSV {history_csv}: {e} — skipping trend plots.",
            file=sys.stderr,
        )
        return
    df = df.dropna(subset=["date"]).sort_values("date")

    if df.empty:
        print("History CSV is empty — no plots generated.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_metric(
        df,
        "pass_rate",
        "Weekly Pass Rate Progression",
        "Pass rate (%)",
        output_dir / "trend_pass_rate.png",
    )
    _plot_metric(
        df,
        "score_mean",
        "Weekly Score Mean Progression",
        "Score mean",
        output_dir / "trend_score_mean.png",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Collect run summaries, append to history CSV, and regenerate trend plots."""
    args = _parse_args()
    all_rows: list[dict] = []

    for suite, json_path in zip(args.suites, args.jsons):
        if not json_path.exists():
            print(
                f"WARNING: {json_path} not found, skipping suite {suite!r}",
                file=sys.stderr,
            )
            continue
        with open(json_path, encoding="utf-8") as fh:
            summary = json.load(fh)
        date = _run_date(summary, args.date)
        rows = _rows_from_summary(summary, suite, date)
        all_rows.extend(rows)
        print(f"Collected {len(rows)} row(s) for suite={suite!r} date={date!r}")

    if all_rows:
        _append_to_history(args.history_csv, all_rows)
        print(f"Appended {len(all_rows)} total row(s) to {args.history_csv}")

    try:
        _generate_plots(args.history_csv, args.output_dir)
    except Exception as e:
        print(
            f"WARNING: trend plot step failed (non-fatal): {e}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
