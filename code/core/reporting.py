"""Reporting utilities shared by all modeling strategies.

This module keeps execution logging, submission writing and leaderboard creation
outside the model files. The strategy modules only need to return a
``StrategyResult`` object with their selected clients, selected variables and
validation estimate.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class StrategyResult:
    """Standard output returned by every modeling strategy.

    Attributes:
        strategy: Short strategy identifier used in file names and leaderboard rows.
        selected_clients: 1-based test-set row indexes selected for contact.
        used_features: Variable indexes saved to the ``*_vars.txt`` file.
        estimated_profit: Validation or cross-validation profit estimate in EUR.
        opt_k: Number of selected customers chosen on validation data.
        model_label: Human-readable description of the final estimator.
        validation_scheme: Description of how the profit estimate was obtained.
        extra: Optional additional values useful for diagnostics or reporting.
        output_dir: Directory where strategy-specific artifacts are stored.
        log_path: Path to the run log captured from stdout/stderr.
        obs_path: Path to the generated ``*_obs.txt`` submission file.
        vars_path: Path to the generated ``*_vars.txt`` submission file.
    """

    strategy: str
    selected_clients: list[int]
    used_features: list[int]
    estimated_profit: float | None = None
    opt_k: int | None = None
    model_label: str = ""
    validation_scheme: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    output_dir: str | None = None
    log_path: str | None = None
    obs_path: str | None = None
    vars_path: str | None = None

    def __iter__(self) -> Iterator[list[int]]:
        """Allow backward-compatible unpacking into clients and variables."""
        yield self.selected_clients
        yield self.used_features

    @property
    def n_selected_clients(self) -> int:
        """Return the number of test customers selected by the strategy."""
        return len(self.selected_clients)

    @property
    def n_features(self) -> int:
        """Return the number of variables used by the final model."""
        return len(self.used_features)

    @property
    def feature_cost(self) -> int:
        """Return the acquisition cost of the final variable set in EUR."""
        return self.n_features * 200

    def to_row(self) -> dict[str, Any]:
        """Convert the result into a flat dictionary for CSV reporting."""
        row = {
            "strategy": self.strategy,
            "estimated_profit": self.estimated_profit,
            "validation_scheme": self.validation_scheme,
            "model_label": self.model_label,
            "opt_k": self.opt_k,
            "selected_clients": self.n_selected_clients,
            "n_features": self.n_features,
            "feature_cost_eur": self.feature_cost,
            "output_dir": self.output_dir,
            "log_path": self.log_path,
            "obs_path": self.obs_path,
            "vars_path": self.vars_path,
            "used_features": ",".join(map(str, self.used_features)),
        }
        for key, value in self.extra.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[f"extra_{key}"] = value
        return row


class TeeStream:
    """Write every message to both the terminal and a log file."""

    def __init__(self, terminal: TextIO, log_file: TextIO) -> None:
        """Store the terminal stream and the opened log file handle."""
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message: str) -> int:
        """Write a message to both destinations and return its length."""
        self.terminal.write(message)
        self.log_file.write(message)
        return len(message)

    def flush(self) -> None:
        """Flush both output streams."""
        self.terminal.flush()
        self.log_file.flush()


@contextlib.contextmanager
def tee_output(log_path: str | os.PathLike[str]) -> Iterator[None]:
    """Mirror stdout and stderr to a run log while keeping terminal output visible."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = TeeStream(old_stdout, log_file)
        sys.stderr = TeeStream(old_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def coerce_strategy_result(
    raw_output: StrategyResult | tuple[list[int], list[int]],
    strategy: str,
) -> StrategyResult:
    """Convert legacy tuple outputs into ``StrategyResult`` objects if needed."""
    if isinstance(raw_output, StrategyResult):
        return raw_output

    selected_clients, used_features = raw_output
    return StrategyResult(
        strategy=strategy,
        selected_clients=list(selected_clients),
        used_features=list(used_features),
        validation_scheme="not reported",
    )


def save_submission_files(
    result: StrategyResult,
    output_dir: str | os.PathLike[str],
    student_ids: str,
) -> StrategyResult:
    """Write ``*_obs.txt`` and ``*_vars.txt`` files for a completed strategy."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    obs_path = output / f"{student_ids}_obs.txt"
    vars_path = output / f"{student_ids}_vars.txt"

    with open(obs_path, "w", encoding="utf-8") as file:
        for client_id in result.selected_clients[:1000]:
            file.write(f"{int(client_id)}\n")

    with open(vars_path, "w", encoding="utf-8") as file:
        for variable_id in result.used_features:
            file.write(f"{int(variable_id)}\n")

    result.output_dir = str(output)
    result.obs_path = str(obs_path)
    result.vars_path = str(vars_path)
    return result


def save_strategy_summary(
    result: StrategyResult,
    output_dir: str | os.PathLike[str],
    filename: str | None = None,
) -> str:
    """Save a one-row CSV summary for one strategy and return the file path."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / (filename or f"{result.strategy}_summary.csv")
    pd.DataFrame([result.to_row()]).to_csv(path, index=False)
    return str(path)


def build_leaderboard(results: Iterable[StrategyResult]) -> pd.DataFrame:
    """Create a sorted leaderboard DataFrame from strategy results."""
    df = pd.DataFrame([result.to_row() for result in results])
    if df.empty:
        return df

    df["estimated_profit_sort"] = df["estimated_profit"].fillna(-np.inf)
    df = df.sort_values(
        ["estimated_profit_sort", "n_features", "selected_clients"],
        ascending=[False, True, True],
    ).drop(columns=["estimated_profit_sort"])
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def save_leaderboard_csv(
    results: Iterable[StrategyResult],
    output_dir: str | os.PathLike[str],
    filename: str = "strategy_leaderboard.csv",
) -> tuple[pd.DataFrame, str]:
    """Save the strategy leaderboard to CSV and return both DataFrame and path."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    leaderboard = build_leaderboard(results)
    path = output / filename
    leaderboard.to_csv(path, index=False)
    return leaderboard, str(path)


def plot_leaderboard(
    leaderboard: pd.DataFrame,
    output_dir: str | os.PathLike[str],
    filename: str = "strategy_leaderboard.png",
) -> str | None:
    """Plot estimated profit by strategy and save the leaderboard figure."""
    if leaderboard.empty or "estimated_profit" not in leaderboard:
        return None

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = leaderboard.sort_values("estimated_profit", ascending=True).copy()

    labels = [
        f"{row.strategy}\n{int(row.n_features)} vars, K={int(row.selected_clients)}"
        for row in df.itertuples(index=False)
    ]

    plt.figure(figsize=(10, max(4, 0.85 * len(df))))
    y_pos = np.arange(len(df))
    plt.barh(y_pos, df["estimated_profit"].astype(float))
    plt.yticks(y_pos, labels)
    plt.xlabel("Estimated validation profit [EUR]")
    plt.title("Strategy leaderboard")
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    for position, value in zip(y_pos, df["estimated_profit"].astype(float)):
        plt.text(value, position, f" {value:.0f}", va="center")

    path = output / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return str(path)
