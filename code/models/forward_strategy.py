"""Wrapper for the greedy forward-selection baseline."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from core.metrics import optimize_top_k
from core.reporting import StrategyResult, save_strategy_summary
from models.forward_selection import forward_selection

RANDOM_STATE = 42
MAX_CLIENTS = 1000
VARIABLE_COST = 200


def _feature_to_submission_index(feature: str, columns: pd.Index) -> int:
    """Convert a feature name to the integer id required by the submission file."""
    text = str(feature)
    if text.upper().startswith("V") and text[1:].isdigit():
        return int(text[1:])
    return list(columns).index(feature) + 1


def _save_test_probability_plot(
    probabilities: np.ndarray,
    cutoff_probability: float,
    n_select: int,
    output_dir: str,
) -> str:
    """Save a histogram of test probabilities with the selected cutoff."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.hist(probabilities, bins=40, alpha=0.7)
    plt.axvline(cutoff_probability, linestyle="--", label=f"Top-{n_select} cutoff")
    plt.xlabel("Predicted probability on test set")
    plt.ylabel("Number of clients")
    plt.title("Forward selection: test probability distribution")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    path = os.path.join(output_dir, "forward_test_probability_distribution.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def _evaluate_holdout_profit(
    model: GradientBoostingClassifier,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    selected_features: list[str],
) -> tuple[float, int]:
    """Compute top-K project profit for the selected features on the holdout set."""
    valid_probs = model.predict_proba(X_valid[selected_features])[:, 1]
    return optimize_top_k(y_valid.values, valid_probs, len(selected_features), MAX_CLIENTS)


def run_forward(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    output_dir: str = ".",
) -> StrategyResult:
    """Run the forward-selection baseline and return a standard result.

    This method uses a single stratified train/validation split. It is kept as a
    simple baseline and as an additional candidate source for the combined model;
    it should not be described as a full cross-validated strategy.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Forward Selection Strategy")
    print("=" * 65)

    base_model = GradientBoostingClassifier(n_estimators=10, random_state=RANDOM_STATE)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.30,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    selected_idx, best_threshold, best_candidates = forward_selection(
        X_tr.values,
        y_tr.values,
        X_val.values,
        y_val.values,
        model=base_model,
    )
    selected_idx = [int(index) for index in selected_idx]
    selected_features = [X_train.columns[index] for index in selected_idx]
    if not selected_features:
        raise RuntimeError("Forward selection returned no features.")

    print(f"Selected features : {selected_features}")
    print(f"Best threshold    : {best_threshold}")
    print(f"Best K candidates : {best_candidates}")

    validation_model = clone(base_model)
    validation_model.fit(X_tr[selected_features], y_tr)
    holdout_profit, holdout_k = _evaluate_holdout_profit(validation_model, X_val, y_val, selected_features)
    print(f"Holdout profit    : {holdout_profit:.1f} EUR")
    print(f"Holdout K         : {holdout_k} clients")

    final_model = clone(base_model)
    final_model.fit(X_train[selected_features], y_train)
    test_probs = final_model.predict_proba(X_test[selected_features])[:, 1]

    n_select = min(int(best_candidates), MAX_CLIENTS)
    top_indices = np.argsort(test_probs)[::-1][:n_select]
    cutoff_probability = float(test_probs[top_indices[-1]]) if len(top_indices) else 0.0
    plot_path = _save_test_probability_plot(test_probs, cutoff_probability, n_select, output_dir)
    print(f"Saved plot: {plot_path}")

    selected_clients = [int(index) + 1 for index in top_indices]
    used_features = [_feature_to_submission_index(feature, X_train.columns) for feature in selected_features]

    print(f"\nSelected clients : {len(selected_clients)}")
    print(f"Used features    : {len(used_features)}")
    print(f"Feature cost     : {len(used_features) * VARIABLE_COST} EUR")

    result = StrategyResult(
        strategy="forward",
        selected_clients=selected_clients,
        used_features=used_features,
        estimated_profit=holdout_profit,
        opt_k=holdout_k,
        model_label="gradient_boosting_forward_selection",
        validation_scheme="single 70/30 stratified holdout",
        extra={
            "feature_names": ",".join(selected_features),
            "selection_threshold": float(best_threshold),
            "selection_k": int(best_candidates),
        },
    )
    summary_path = save_strategy_summary(result, output_dir, "forward_summary.csv")
    print(f"Saved summary: {summary_path}")
    return result
