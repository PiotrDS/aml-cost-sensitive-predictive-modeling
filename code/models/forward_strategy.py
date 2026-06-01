"""
Standalone wrapper for forward_selection.py.

It turns the existing forward_selection() helper into a full submission strategy
returning selected test clients and used variables.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from models.forward_selection import forward_selection


RANDOM_STATE = 42
MAX_CLIENTS = 1000



def _feature_to_submission_index(feature: str, columns) -> int:
    text = str(feature)
    if text.upper().startswith("V") and text[1:].isdigit():
        return int(text[1:])
    return list(columns).index(feature) + 1



def run_forward(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    output_dir: str = ".",
) -> tuple[list[int], list[int]]:
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("  Forward Selection Strategy")
    print("=" * 65)

    base_model = GradientBoostingClassifier(
        n_estimators=10,
        random_state=RANDOM_STATE,
    )

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

    selected_idx = [int(i) for i in selected_idx]
    selected_features = [X_train.columns[i] for i in selected_idx]

    if len(selected_features) == 0:
        raise RuntimeError("Forward selection returned no features.")

    print(f"Selected features : {selected_features}")
    print(f"Best threshold    : {best_threshold}")
    print(f"Best K candidates : {best_candidates}")

    final_model = clone(base_model)
    final_model.fit(X_train[selected_features], y_train)
    test_probs = final_model.predict_proba(X_test[selected_features])[:, 1]

    n_select = min(int(best_candidates), MAX_CLIENTS)
    top_indices = np.argsort(test_probs)[::-1][:n_select]

    # Simple plot for report/presentation: selected test-score distribution.
    plt.figure(figsize=(9, 5))
    plt.hist(test_probs, bins=40, alpha=0.7)
    plt.axvline(
        test_probs[top_indices[-1]],
        linestyle="--",
        label=f"Top-{n_select} cutoff",
    )
    plt.xlabel("Predicted probability on test set")
    plt.ylabel("Number of clients")
    plt.title("Forward selection: test probability distribution")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plot_path = os.path.join(output_dir, "forward_test_probability_distribution.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {plot_path}")

    best_clients_1_based = [int(i) + 1 for i in top_indices]
    used_features_idx = [
        _feature_to_submission_index(feature, X_train.columns)
        for feature in selected_features
    ]

    print(f"\nSelected clients : {len(best_clients_1_based)}")
    print(f"Used features    : {len(used_features_idx)}")
    print(f"Feature cost     : {len(used_features_idx) * 200} EUR")

    return best_clients_1_based, used_features_idx
