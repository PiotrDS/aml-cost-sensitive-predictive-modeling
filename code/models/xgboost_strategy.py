"""XGBoost strategy optimized under the project profit function."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

from core.metrics import optimize_top_k
from core.plotting import (
    plot_feature_importance,
    plot_probability_distribution,
    plot_profit_optimization_curve,
)
from core.reporting import StrategyResult, save_strategy_summary

RANDOM_STATE = 42
MAX_CLIENTS = 1000
VARIABLE_COST = 200


XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.1,
    "scale_pos_weight": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "eval_metric": "logloss",
}


def _feature_to_submission_index(feature: str, columns: pd.Index) -> int:
    """Convert a variable name like ``V123`` into the submission variable id."""
    text = str(feature)
    if text.upper().startswith("V") and text[1:].isdigit():
        return int(text[1:])
    return list(columns).index(feature) + 1


def _rank_features_by_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.Series:
    """Fit one XGBoost model and return variables sorted by importance."""
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)
    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    if (importances > 0).any():
        importances = importances[importances > 0]
    return importances


def _cross_validated_probabilities(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: list[str],
    cv: StratifiedKFold,
) -> np.ndarray:
    """Return out-of-fold XGBoost probabilities for a fixed feature set."""
    oof_probs = np.zeros(len(X_train))
    for train_idx, valid_idx in cv.split(X_train, y_train):
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train.iloc[train_idx][features], y_train.iloc[train_idx])
        oof_probs[valid_idx] = model.predict_proba(X_train.iloc[valid_idx][features])[:, 1]
    return oof_probs


def _sweep_feature_count(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    importances: pd.Series,
    cv: StratifiedKFold,
    max_features: int = 30,
) -> tuple[int, float, pd.DataFrame]:
    """Evaluate top-N XGBoost feature sets and return the best feature count."""
    best_profit = -np.inf
    best_num_features = 1
    rows = []

    for num_features in range(1, min(max_features, len(importances)) + 1):
        current_features = importances.index[:num_features].tolist()
        oof_probs = _cross_validated_probabilities(X_train, y_train, current_features, cv)
        profit, opt_k = optimize_top_k(y_train.values, oof_probs, num_features, MAX_CLIENTS)
        rows.append({"n_features": num_features, "cv_profit": profit, "opt_k": opt_k})

        print(f"  Features: {num_features:2d} | CV Profit: {profit:8.1f} EUR  (opt. K={opt_k})")
        if profit > best_profit:
            best_profit = profit
            best_num_features = num_features

    return best_num_features, float(best_profit), pd.DataFrame(rows)


def run_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    output_dir: str = ".",
) -> StrategyResult:
    """Train the XGBoost strategy and return selected test clients and variables.

    The method ranks variables with XGBoost, sweeps the number of top variables
    using 5-fold out-of-fold predictions, chooses the best top-K contact cutoff,
    and finally refits the selected model on the full training set.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Initializing XGBoost strategy...")
    importances = _rank_features_by_xgboost(X_train, y_train)

    print("\n--- Feature Optimization (5-Fold CV, top-K evaluation) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_num_features, best_sweep_profit, sweep_df = _sweep_feature_count(X_train, y_train, importances, cv)

    sweep_path = os.path.join(output_dir, "xgboost_feature_count_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False)
    print(f"Saved sweep table: {sweep_path}")

    profit_plot = plot_profit_optimization_curve(
        sweep_df["n_features"],
        sweep_df["cv_profit"],
        x_label="Number of selected features",
        title="XGBoost profit vs number of variables",
        best_x=best_num_features,
        output_dir=output_dir,
        filename="xgboost_profit_optimization.png",
    )
    print(f"Saved plot: {profit_plot}")

    importance_plot = plot_feature_importance(
        importances.index.tolist(),
        importances.values,
        "XGBoost initial feature importance",
        output_dir,
        "xgboost_feature_importance.png",
        top_n=20,
    )
    print(f"Saved plot: {importance_plot}")

    selected_features = importances.index[:best_num_features].tolist()
    print("-" * 55)
    print(f"  Optimal features : {best_num_features}  (CV profit: {best_sweep_profit:.1f} EUR)")
    print(f"  Selected features: {selected_features}\n")

    oof_probs = _cross_validated_probabilities(X_train, y_train, selected_features, cv)
    final_cv_profit, final_k = optimize_top_k(y_train.values, oof_probs, best_num_features, MAX_CLIENTS)
    print(f"  Final K          : {final_k} clients")
    print(f"  Final CV profit  : {final_cv_profit:.1f} EUR")

    probability_plot = plot_probability_distribution(
        y_train.values,
        oof_probs,
        f"XGBoost OOF probabilities ({best_num_features} variables)",
        output_dir,
        "xgboost_prob_distribution.png",
    )
    print(f"Saved plot: {probability_plot}")

    final_model = xgb.XGBClassifier(**XGB_PARAMS)
    final_model.fit(X_train[selected_features], y_train)
    test_probs = final_model.predict_proba(X_test[selected_features])[:, 1]
    n_select = min(final_k, MAX_CLIENTS)
    top_indices = np.argsort(test_probs)[::-1][:n_select]

    selected_clients = [int(index) + 1 for index in top_indices]
    used_features = [_feature_to_submission_index(feature, X_train.columns) for feature in selected_features]

    print(f"\n  Selected clients : {len(selected_clients)}")
    print(f"  Used features    : {len(used_features)}")
    print(f"  Feature cost     : {len(used_features) * VARIABLE_COST} EUR")

    result = StrategyResult(
        strategy="xgboost",
        selected_clients=selected_clients,
        used_features=used_features,
        estimated_profit=final_cv_profit,
        opt_k=final_k,
        model_label=f"xgboost_top_{best_num_features}",
        validation_scheme="5-fold OOF CV",
        extra={
            "feature_names": ",".join(selected_features),
            "sweep_best_profit": best_sweep_profit,
            "sweep_csv": sweep_path,
        },
    )
    summary_path = save_strategy_summary(result, output_dir, "xgboost_summary.csv")
    print(f"Saved summary: {summary_path}")
    return result
