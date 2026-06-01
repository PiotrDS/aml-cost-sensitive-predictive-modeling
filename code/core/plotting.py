from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("muted")


def _prepare_output_path(output_dir: str | os.PathLike[str], filename: str) -> str:
    """Create the output directory and return a full file path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return str(Path(output_dir) / filename)


def plot_feature_importance(
    features: Sequence[str],
    importances: Sequence[float],
    title: str,
    output_dir: str | os.PathLike[str],
    filename: str = "feature_importance.png",
    top_n: int = 20,
) -> str | None:
    """Save a horizontal bar plot of the most important variables.

    Args:
        features: Feature names aligned with ``importances``.
        importances: Importance values or model coefficients.
        title: Figure title.
        output_dir: Directory where the figure is written.
        filename: Name of the output PNG file.
        top_n: Maximum number of variables shown.

    Returns:
        Path to the saved figure, or ``None`` when there are no features.
    """
    if len(features) == 0:
        return None

    values = np.asarray(importances)
    order = np.argsort(np.abs(values))[::-1][:top_n]
    selected_features = [features[i] for i in order]
    selected_values = [values[i] for i in order]

    plt.figure(figsize=(10, max(6, int(len(selected_features) * 0.3))))
    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in selected_values]
    y_pos = np.arange(len(selected_features))

    plt.barh(y_pos, selected_values, color=colors, edgecolor="black", alpha=0.8)
    plt.yticks(y_pos, selected_features)
    plt.gca().invert_yaxis()
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Importance / coefficient", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    path = _prepare_output_path(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_probability_distribution(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    title: str,
    output_dir: str | os.PathLike[str],
    filename: str = "prob_distribution.png",
) -> str:
    """Save a class-conditional distribution plot of predicted probabilities."""
    y_true = np.asarray(y_true).ravel()
    probabilities = np.asarray(probabilities).ravel()

    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        probabilities[y_true == 0],
        fill=True,
        color="#d62728",
        label="Class 0 (no conversion)",
        alpha=0.4,
    )
    sns.kdeplot(
        probabilities[y_true == 1],
        fill=True,
        color="#2ca02c",
        label="Class 1 (conversion)",
        alpha=0.4,
    )

    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Predicted probability P(Y=1)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(loc="upper right")
    plt.xlim(0, 1)

    path = _prepare_output_path(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_profit_optimization_curve(
    x_values: Sequence[float],
    profit_values: Sequence[float],
    x_label: str,
    title: str,
    best_x: float,
    output_dir: str | os.PathLike[str],
    filename: str = "profit_optimization.png",
) -> str:
    """Save a profit curve and mark the best selected value."""
    profit_values = list(profit_values)

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, profit_values, marker="o", linestyle="-", linewidth=2, markersize=6)
    plt.axvline(x=best_x, color="red", linestyle="--", linewidth=2, label=f"Best: {best_x}")

    best_profit = max(profit_values)
    plt.axhline(
        y=best_profit,
        color="green",
        linestyle=":",
        linewidth=1.5,
        alpha=0.6,
        label=f"Max profit: {best_profit:.1f} EUR",
    )

    plt.title(title, fontsize=14, pad=15)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel("Estimated validation profit [EUR]", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    path = _prepare_output_path(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_lasso_sweep(
    c_values: Sequence[float],
    profit_values: Sequence[float],
    features_count: Sequence[int],
    best_c: float,
    output_dir: str | os.PathLike[str],
    filename: str = "lasso_sweep.png",
) -> str:
    """Save the LASSO regularization sweep plot used in the report."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(c_values, profit_values, marker="o", lw=2, markersize=5)
    ax1.axvline(best_c, color="red", ls="--", label=f"Best C={best_c:.4f}")
    ax1.set_ylabel("CV profit [EUR]", fontsize=11)
    ax1.set_title("Profit vs L1 regularization strength", fontsize=13, pad=10)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(c_values, features_count, marker="s", lw=2, markersize=5)
    ax2.axvline(best_c, color="red", ls="--")
    ax2.set_xlabel("C (inverse regularization strength)", fontsize=11)
    ax2.set_ylabel("Average active features", fontsize=11)
    ax2.set_xscale("log")
    ax2.grid(True, linestyle="--", alpha=0.6)

    path = _prepare_output_path(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path
