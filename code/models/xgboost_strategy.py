import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold


def _optimize_top_k(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_vars: int,
    max_clients: int = 1000,
) -> tuple[float, int]:
    """
    Vectorised search for K in [1, max_clients] that maximises profit.
    Clients are sorted descending by predicted probability so the first K
    rows are exactly the ones we would contact.
    """
    order = np.argsort(probs)[::-1]
    y_sorted = y_true[order[:max_clients]]

    cum_tp = np.cumsum(y_sorted == 1)
    cum_fp = np.cumsum(y_sorted == 0)
    ks = np.arange(1, len(y_sorted) + 1)

    profits = cum_tp * 10 - cum_fp * 5 - n_vars * 200
    best = int(np.argmax(profits))
    return float(profits[best]), int(ks[best])


def run_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    output_dir: str = ".",
) -> tuple[list[int], list[int]]:
    print("Initializing XGBoost strategy...")

    xgb_params = {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "scale_pos_weight": 2,
        "random_state": 42,
        "n_jobs": -1,
    }

    # ── Step 1: rank features by importance on full training data ──────────
    model_full = xgb.XGBClassifier(**xgb_params)
    model_full.fit(X_train, y_train)

    importances = pd.Series(
        model_full.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)
    if (importances > 0).any():
        importances = importances[importances > 0]

    # ── Step 2: CV feature-count sweep – use top-K profit, not a threshold ─
    print("\n--- Feature Optimization (5-Fold CV, top-K evaluation) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_profit = -np.inf
    best_num_features = 1
    profit_ranking = []

    for k in range(1, min(30, len(importances)) + 1):
        current_features = importances.index[:k].tolist()
        oof_probs = np.zeros(len(X_train))

        for tr_idx, val_idx in cv.split(X_train, y_train):
            m = xgb.XGBClassifier(**xgb_params)
            m.fit(
                X_train.iloc[tr_idx][current_features],
                y_train.iloc[tr_idx],
            )
            oof_probs[val_idx] = m.predict_proba(
                X_train.iloc[val_idx][current_features]
            )[:, 1]

        profit, opt_k = _optimize_top_k(y_train.values, oof_probs, n_vars=k)
        profit_ranking.append((k, profit))
        print(f"  Features: {k:2d} | CV Profit: {profit:8.1f} EUR  (opt. K={opt_k})")

        if profit > best_profit:
            best_profit = profit
            best_num_features = k

    from core.plotting import (
        plot_feature_importance,
        plot_probability_distribution,
        plot_profit_optimization_curve,
    )

    # ── Plot ───────────────────────────────────────────────────────────────
    features, profits = zip(*profit_ranking)
    plot_profit_optimization_curve(
        features,
        profits,
        x_label="Number of Selected Features (Top K)",
        title="Profit Optimization Curve vs Number of Features (top-K evaluation)",
        best_x=best_num_features,
        output_dir=output_dir,
        filename="xgboost_profit_optimization.png",
    )
    print(f"Saved plot: {os.path.join(output_dir, 'xgboost_profit_optimization.png')}")

    plot_feature_importance(
        importances.index.tolist(),
        importances.values,
        "XGBoost Initial Feature Importance",
        output_dir,
        "xgboost_feature_importance.png",
        top_n=20,
    )
    print(f"Saved plot: {os.path.join(output_dir, 'xgboost_feature_importance.png')}")

    top_features = importances.index[:best_num_features].tolist()
    print("-" * 55)
    print(
        f"  Optimal features : {best_num_features}  (CV profit: {best_profit:.1f} EUR)"
    )
    print(f"  Selected features: {top_features}\n")

    # ── Step 3: OOF probs on selected features → find final K ─────────────
    oof_probs_final = np.zeros(len(X_train))
    for tr_idx, val_idx in cv.split(X_train[top_features], y_train):
        m = xgb.XGBClassifier(**xgb_params)
        m.fit(X_train.iloc[tr_idx][top_features], y_train.iloc[tr_idx])
        oof_probs_final[val_idx] = m.predict_proba(X_train.iloc[val_idx][top_features])[
            :, 1
        ]

    final_cv_profit, final_k = _optimize_top_k(
        y_train.values, oof_probs_final, n_vars=best_num_features
    )

    print(f"  Final K          : {final_k} clients")
    print(f"  Final CV profit  : {final_cv_profit:.1f} EUR")

    plot_probability_distribution(
        y_train.values,
        oof_probs_final,
        f"XGBoost OOF Probabilities (Final {best_num_features} Features)",
        output_dir,
        "xgboost_prob_distribution.png",
    )
    print(f"Saved plot: {os.path.join(output_dir, 'xgboost_prob_distribution.png')}")

    # ── Step 4: retrain on all data, predict test set, take top final_k ───
    final_model = xgb.XGBClassifier(**xgb_params)
    final_model.fit(X_train[top_features], y_train)

    test_probs = final_model.predict_proba(X_test[top_features])[:, 1]
    n_select = min(final_k, 1000)
    top_indices = np.argsort(test_probs)[::-1][:n_select]

    best_clients_1_based = [int(i) + 1 for i in top_indices]
    used_features_idx = [int(v.replace("V", "")) for v in top_features]

    print(f"\n  Selected clients : {len(best_clients_1_based)}")
    print(f"  Used features    : {len(used_features_idx)}")
    print(f"  Feature cost     : {len(used_features_idx) * 200} EUR")

    return best_clients_1_based, used_features_idx
