"""
Combined cost-sensitive strategy for AML Project 2.

Idea
----
Use the three separately developed strategies as complementary feature selectors:
  1. LASSO / L1 logistic regression -> sparse linear signals
  2. XGBoost feature importance      -> nonlinear signals and interactions
  3. Forward selection               -> greedy profit-driven feature search

Then combine only the candidate variables, prune them under the project profit
function, and compare Logistic Regression, XGBoost and a weighted ensemble on the
same final feature set.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from core.reporting import StrategyResult, save_strategy_summary

try:
    from models.forward_selection import forward_selection
except Exception:
    forward_selection = None


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_STATE = 42
MAX_CLIENTS = 1000
VARIABLE_COST = 200

XGB_PARAMS = {
    "n_estimators": 80,
    "max_depth": 3,
    "learning_rate": 0.08,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "scale_pos_weight": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "eval_metric": "logloss",
}


@dataclass
class EvaluationResult:
    """Cross-validated evaluation summary for one feature set."""

    best_profit: float
    best_k: int
    best_model: str
    best_weight_xgb: float
    best_probs: np.ndarray
    all_rows: list[dict]
    ks: np.ndarray
    profits_by_k: np.ndarray


# ---------------------------------------------------------------------
# Profit helpers
# ---------------------------------------------------------------------


def _as_numpy_y(y_train: pd.Series | np.ndarray) -> np.ndarray:
    """Return the target vector as a one-dimensional NumPy array."""
    return np.asarray(y_train).ravel()



def _profit_curve_top_k(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_vars: int,
    max_clients: int = MAX_CLIENTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return profit for K=1..max_clients after sorting by probability."""
    limit = min(max_clients, len(y_true))
    order = np.argsort(probs)[::-1]
    y_sorted = y_true[order[:limit]]

    cum_tp = np.cumsum(y_sorted == 1)
    cum_fp = np.cumsum(y_sorted == 0)
    ks = np.arange(1, limit + 1)
    profits = cum_tp * 10 - cum_fp * 5 - n_vars * VARIABLE_COST
    return ks, profits.astype(float)



def _optimize_top_k(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_vars: int,
    max_clients: int = MAX_CLIENTS,
) -> tuple[float, int, np.ndarray, np.ndarray]:
    """Return the best profit, best K and full top-K profit curve."""
    ks, profits = _profit_curve_top_k(y_true, probs, n_vars, max_clients)
    best_idx = int(np.argmax(profits))
    return float(profits[best_idx]), int(ks[best_idx]), ks, profits


# ---------------------------------------------------------------------
# Feature candidate generation
# ---------------------------------------------------------------------


def _xgb_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int = 30,
) -> tuple[list[str], dict[str, float]]:
    """Top variables by XGBoost gain/importance on the full training set."""
    print(f"[1/5] XGBoost candidate search: top {top_n} features")

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)
    importances = importances[importances > 0]
    selected = importances.head(top_n)

    return selected.index.tolist(), selected.to_dict()



def _lasso_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    mi_pool_size: int = 150,
    top_n: int = 30,
) -> tuple[list[str], dict[str, float], pd.DataFrame]:
    """
    Candidate variables from L1 logistic regression.

    The full-data MI screening is used here only to build a candidate pool for
    the combined method. The final comparison still uses OOF predictions.
    """
    print(f"[2/5] LASSO candidate search: MI pool {mi_pool_size}, keep up to {top_n}")

    y = _as_numpy_y(y_train)
    col_names = X_train.columns.tolist()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi = mutual_info_classif(X_train, y, random_state=RANDOM_STATE)

    mi_idx = np.argsort(mi)[::-1][: min(mi_pool_size, X_train.shape[1])]
    mi_features = [col_names[i] for i in mi_idx]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    c_values = np.logspace(-2.5, 0.7, 12)
    sweep_rows = []

    for C in c_values:
        oof = np.zeros(len(y))
        nonzero_counts = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for tr_idx, val_idx in cv.split(X_train[mi_features], y):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train.iloc[tr_idx][mi_features])
                X_val = scaler.transform(X_train.iloc[val_idx][mi_features])

                model = LogisticRegression(
                    penalty="l1",
                    C=float(C),
                    solver="saga",
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )
                model.fit(X_tr, y[tr_idx])
                oof[val_idx] = model.predict_proba(X_val)[:, 1]
                nonzero_counts.append(int(np.sum(model.coef_[0] != 0)))

        avg_n = int(round(np.mean(nonzero_counts)))
        if avg_n == 0:
            profit, k = -np.inf, 0
        else:
            profit, k, _, _ = _optimize_top_k(y, oof, avg_n)

        sweep_rows.append(
            {"C": float(C), "avg_features": avg_n, "cv_profit": profit, "opt_k": k}
        )

    sweep_df = pd.DataFrame(sweep_rows)
    valid = sweep_df[sweep_df["avg_features"] > 0].copy()
    if valid.empty:
        print("      WARNING: LASSO produced no active features. Skipping LASSO source.")
        return [], {}, sweep_df

    best_C = float(valid.sort_values("cv_profit", ascending=False).iloc[0]["C"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train[mi_features])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_lasso = LogisticRegression(
            penalty="l1",
            C=best_C,
            solver="saga",
            max_iter=3000,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        final_lasso.fit(X_scaled, y)

    coef = pd.Series(np.abs(final_lasso.coef_[0]), index=mi_features)
    coef = coef[coef > 0].sort_values(ascending=False).head(top_n)

    print(
        f"      Best C={best_C:.4f}; selected {len(coef)} LASSO candidates "
        f"from CV-guided sparse model"
    )

    return coef.index.tolist(), coef.to_dict(), sweep_df



def _forward_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int = 20,
) -> tuple[list[str], dict[str, float]]:
    """Candidate variables from the existing forward_selection.py strategy."""
    print(f"[3/5] Forward-selection candidate search: keep up to {top_n}")

    if forward_selection is None:
        print("      WARNING: models.forward_selection could not be imported. Skipping.")
        return [], {}

    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.30,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )

        selected_idx, _, _ = forward_selection(
            X_tr.values,
            y_tr.values,
            X_val.values,
            y_val.values,
            model=GradientBoostingClassifier(
                n_estimators=10,
                random_state=RANDOM_STATE,
            ),
        )

        selected_idx = list(selected_idx)[:top_n]
        features = [X_train.columns[int(i)] for i in selected_idx]

        # Higher score for earlier greedy selections.
        scores = {f: float(top_n - rank) for rank, f in enumerate(features)}
        return features, scores

    except Exception as exc:
        print(f"      WARNING: forward selection failed: {exc}. Skipping this source.")
        return [], {}



def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Scale source-specific scores to the [0, 1] range."""
    if not scores:
        return {}
    max_abs = max(abs(v) for v in scores.values())
    if max_abs == 0:
        return {k: 0.0 for k in scores}
    return {k: float(abs(v) / max_abs) for k, v in scores.items()}



def _build_combined_candidate_pool(
    source_features: dict[str, list[str]],
    source_scores: dict[str, dict[str, float]],
    max_candidates: int = 22,
) -> tuple[list[str], pd.DataFrame]:
    """Combine sources but prioritize agreement and strong within-method rank."""
    all_features = sorted(set().union(*[set(v) for v in source_features.values()]))
    norm_scores = {src: _normalize_scores(sc) for src, sc in source_scores.items()}

    rows = []
    for feature in all_features:
        sources = [src for src, feats in source_features.items() if feature in feats]
        agreement = len(sources)
        strength = sum(norm_scores.get(src, {}).get(feature, 0.0) for src in source_features)

        # Agreement is deliberately much more important than one strong ranking.
        combined_score = agreement * 100.0 + strength
        rows.append(
            {
                "feature": feature,
                "sources": ", ".join(sources),
                "n_sources": agreement,
                "source_strength": strength,
                "combined_score": combined_score,
            }
        )

    df = pd.DataFrame(rows).sort_values(
        ["n_sources", "combined_score", "source_strength"],
        ascending=[False, False, False],
    )

    selected = df.head(max_candidates)["feature"].tolist()
    return selected, df


# ---------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------


def _evaluate_feature_set(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    features: list[str],
    cv: StratifiedKFold,
    stage: str,
) -> EvaluationResult:
    """
    Evaluate Logistic Regression, XGBoost and weighted ensembles using OOF profit.

    weight_xgb = 0 means pure logistic regression.
    weight_xgb = 1 means pure XGBoost.
    values in between are probability averages.
    """
    y = _as_numpy_y(y_train)
    n_vars = len(features)

    oof_logit = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for tr_idx, val_idx in cv.split(X_train[features], y):
            X_tr_raw = X_train.iloc[tr_idx][features]
            X_val_raw = X_train.iloc[val_idx][features]

            # Logistic regression branch: scaled data.
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr_raw)
            X_val_scaled = scaler.transform(X_val_raw)

            logit = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
            logit.fit(X_tr_scaled, y[tr_idx])
            oof_logit[val_idx] = logit.predict_proba(X_val_scaled)[:, 1]

            # XGBoost branch: raw data.
            xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
            xgb_model.fit(X_tr_raw, y[tr_idx])
            oof_xgb[val_idx] = xgb_model.predict_proba(X_val_raw)[:, 1]

    rows = []
    best_profit = -np.inf
    best_k = 0
    best_weight = 0.0
    best_model = ""
    best_probs = None
    best_ks = None
    best_curve = None

    for weight_xgb in np.linspace(0.0, 1.0, 11):
        probs = weight_xgb * oof_xgb + (1.0 - weight_xgb) * oof_logit
        profit, k, ks, curve = _optimize_top_k(y, probs, n_vars)

        if weight_xgb == 0:
            model_name = "logistic"
        elif weight_xgb == 1:
            model_name = "xgboost"
        else:
            model_name = f"ensemble_w{weight_xgb:.1f}"

        rows.append(
            {
                "stage": stage,
                "model": model_name,
                "weight_xgb": float(weight_xgb),
                "n_features": n_vars,
                "profit": profit,
                "opt_k": k,
            }
        )

        if profit > best_profit:
            best_profit = profit
            best_k = k
            best_weight = float(weight_xgb)
            best_model = model_name
            best_probs = probs
            best_ks = ks
            best_curve = curve

    assert best_probs is not None
    assert best_ks is not None
    assert best_curve is not None

    return EvaluationResult(
        best_profit=best_profit,
        best_k=best_k,
        best_model=best_model,
        best_weight_xgb=best_weight,
        best_probs=best_probs,
        all_rows=rows,
        ks=best_ks,
        profits_by_k=best_curve,
    )



def _greedy_backward_pruning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    initial_features: list[str],
    cv: StratifiedKFold,
    min_features: int = 1,
) -> tuple[list[str], EvaluationResult, pd.DataFrame]:
    """Remove variables as long as OOF profit improves."""
    print("[4/5] Greedy backward pruning under project profit")

    current = list(initial_features)
    current_eval = _evaluate_feature_set(X_train, y_train, current, cv, stage="initial")
    history = [
        {
            "step": 0,
            "removed_feature": "",
            "n_features": len(current),
            "profit": current_eval.best_profit,
            "opt_k": current_eval.best_k,
            "best_model": current_eval.best_model,
        }
    ]

    print(
        f"      Start: {len(current)} features | profit={current_eval.best_profit:.1f} "
        f"| K={current_eval.best_k} | model={current_eval.best_model}"
    )

    step = 1
    improved = True
    while improved and len(current) > min_features:
        improved = False
        best_trial_eval = current_eval
        best_trial_features = current
        removed_best = None

        for feature in current:
            trial_features = [f for f in current if f != feature]
            trial_eval = _evaluate_feature_set(
                X_train,
                y_train,
                trial_features,
                cv,
                stage=f"prune_step_{step}",
            )

            if trial_eval.best_profit > best_trial_eval.best_profit:
                best_trial_eval = trial_eval
                best_trial_features = trial_features
                removed_best = feature

        if removed_best is not None:
            improved = True
            current = best_trial_features
            current_eval = best_trial_eval
            history.append(
                {
                    "step": step,
                    "removed_feature": removed_best,
                    "n_features": len(current),
                    "profit": current_eval.best_profit,
                    "opt_k": current_eval.best_k,
                    "best_model": current_eval.best_model,
                }
            )
            print(
                f"      Removed {removed_best:>8s} -> {len(current):2d} features "
                f"| profit={current_eval.best_profit:.1f} | K={current_eval.best_k} "
                f"| model={current_eval.best_model}"
            )
            step += 1

    print(
        f"      Final: {len(current)} features | profit={current_eval.best_profit:.1f} "
        f"| K={current_eval.best_k} | model={current_eval.best_model}"
    )

    return current, current_eval, pd.DataFrame(history)


# ---------------------------------------------------------------------
# Final prediction
# ---------------------------------------------------------------------


def _predict_test_probs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    features: list[str],
    weight_xgb: float,
) -> np.ndarray:
    """Fit final full-data models and return weighted test probabilities."""
    y = _as_numpy_y(y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[features])
    X_test_scaled = scaler.transform(X_test[features])

    logit = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    logit.fit(X_train_scaled, y)
    probs_logit = logit.predict_proba(X_test_scaled)[:, 1]

    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(X_train[features], y)
    probs_xgb = xgb_model.predict_proba(X_test[features])[:, 1]

    return weight_xgb * probs_xgb + (1.0 - weight_xgb) * probs_logit



def _feature_to_submission_index(feature: str, columns: Iterable[str]) -> int:
    """Convert V123 -> 123; otherwise use 1-based column position."""
    text = str(feature)
    if text.upper().startswith("V"):
        suffix = text[1:]
        if suffix.isdigit():
            return int(suffix)
    return list(columns).index(feature) + 1


# ---------------------------------------------------------------------
# Plots for report / presentation
# ---------------------------------------------------------------------


def _savefig(path: str) -> None:
    """Save the current Matplotlib figure and close it."""
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      Saved plot: {path}")



def _plot_lasso_sweep(sweep_df: pd.DataFrame, output_dir: str) -> None:
    """Save the LASSO regularization sweep used during candidate generation."""
    if sweep_df.empty:
        return

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(sweep_df["C"], sweep_df["cv_profit"], marker="o", linewidth=2)
    ax1.set_xscale("log")
    ax1.set_xlabel("C (inverse regularization strength)")
    ax1.set_ylabel("OOF profit [EUR]")
    ax1.set_title("LASSO candidate search: profit vs regularization")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(sweep_df["C"], sweep_df["avg_features"], marker="s", linestyle=":")
    ax2.set_ylabel("Average active features")

    _savefig(os.path.join(output_dir, "combined_lasso_sweep.png"))



def _plot_feature_sources(pool_df: pd.DataFrame, selected_features: list[str], output_dir: str) -> None:
    """Save a bar chart showing agreement between feature-selection methods."""
    if pool_df.empty:
        return

    df = pool_df[pool_df["feature"].isin(selected_features)].copy()
    df = df.sort_values("combined_score", ascending=True)

    plt.figure(figsize=(9, max(4, 0.35 * len(df))))
    plt.barh(df["feature"], df["n_sources"])
    plt.xlabel("Number of methods selecting the feature")
    plt.ylabel("Feature")
    plt.title("Combined candidate pool: agreement between methods")
    plt.xlim(0, 3.2)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    _savefig(os.path.join(output_dir, "combined_feature_source_agreement.png"))



def _plot_model_comparison(rows: list[dict], output_dir: str) -> None:
    """Save the model-comparison plot for initial and pruned feature sets."""
    if not rows:
        return

    df = pd.DataFrame(rows)
    labels = df["stage"] + " / " + df["model"]

    plt.figure(figsize=(10, max(4, 0.32 * len(df))))
    y_pos = np.arange(len(df))
    plt.barh(y_pos, df["profit"])
    plt.yticks(y_pos, labels)
    plt.xlabel("OOF profit [EUR]")
    plt.title("Model comparison under project profit")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    _savefig(os.path.join(output_dir, "combined_model_comparison.png"))



def _plot_pruning_history(history_df: pd.DataFrame, output_dir: str) -> None:
    """Save the backward-pruning profit history plot."""
    if history_df.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["n_features"], history_df["profit"], marker="o", linewidth=2)
    plt.gca().invert_xaxis()
    plt.xlabel("Number of variables")
    plt.ylabel("OOF profit [EUR]")
    plt.title("Backward pruning: fewer variables vs profit")
    plt.grid(True, linestyle="--", alpha=0.5)
    _savefig(os.path.join(output_dir, "combined_pruning_curve.png"))



def _plot_topk_profit_curve(eval_result: EvaluationResult, output_dir: str) -> None:
    """Save the final top-K customer-contact profit curve."""
    plt.figure(figsize=(9, 5))
    plt.plot(eval_result.ks, eval_result.profits_by_k, linewidth=2)
    plt.axvline(eval_result.best_k, linestyle="--", label=f"Best K={eval_result.best_k}")
    plt.xlabel("Number of contacted customers K")
    plt.ylabel("OOF profit [EUR]")
    plt.title("Final model: profit curve over contacted customers")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    _savefig(os.path.join(output_dir, "combined_topk_profit_curve.png"))



def _plot_weight_sweep(eval_result: EvaluationResult, output_dir: str) -> None:
    """Save the ensemble weight sweep plot."""
    df = pd.DataFrame(eval_result.all_rows)
    if df.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(df["weight_xgb"], df["profit"], marker="o", linewidth=2)
    plt.axvline(
        eval_result.best_weight_xgb,
        linestyle="--",
        label=f"Best weight={eval_result.best_weight_xgb:.1f}",
    )
    plt.xlabel("XGBoost weight in probability average")
    plt.ylabel("OOF profit [EUR]")
    plt.title("Ensemble weight sweep")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    _savefig(os.path.join(output_dir, "combined_weight_sweep.png"))



def _plot_probability_distribution(
    y_true: np.ndarray,
    probs: np.ndarray,
    output_dir: str,
) -> None:
    """Save the OOF probability distribution for the final combined model."""
    plt.figure(figsize=(9, 5))
    plt.hist(probs[y_true == 0], bins=30, alpha=0.55, density=True, label="Class 0")
    plt.hist(probs[y_true == 1], bins=30, alpha=0.55, density=True, label="Class 1")
    plt.xlabel("OOF predicted probability")
    plt.ylabel("Density")
    plt.title("Final model: OOF probability distribution by class")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    _savefig(os.path.join(output_dir, "combined_probability_distribution.png"))


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------


def run_combined(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    output_dir: str = ".",
) -> StrategyResult:
    """Run the combined strategy and return a standard result.

    The method uses XGBoost, LASSO and forward selection as candidate-feature
    sources. It then prunes the combined feature set under 5-fold OOF profit and
    refits the selected weighted ensemble on the full training set.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  Combined Strategy: LASSO + XGBoost + Forward Selection")
    print("=" * 70)

    # 1) Generate candidate features from the three methods.
    xgb_feats, xgb_scores = _xgb_candidates(X_train, y_train, top_n=30)
    lasso_feats, lasso_scores, lasso_sweep = _lasso_candidates(
        X_train,
        y_train,
        mi_pool_size=150,
        top_n=30,
    )
    forward_feats, forward_scores = _forward_candidates(X_train, y_train, top_n=20)

    source_features = {
        "xgboost": xgb_feats,
        "lasso": lasso_feats,
        "forward": forward_feats,
    }
    source_scores = {
        "xgboost": xgb_scores,
        "lasso": lasso_scores,
        "forward": forward_scores,
    }

    candidate_features, pool_df = _build_combined_candidate_pool(
        source_features,
        source_scores,
        max_candidates=22,
    )

    if len(candidate_features) == 0:
        raise RuntimeError("Combined strategy found no candidate features.")

    pool_path = os.path.join(output_dir, "combined_candidate_pool.csv")
    pool_df.to_csv(pool_path, index=False)
    print(f"      Saved candidate pool: {pool_path}")

    print("\nCandidate features selected for pruning:")
    print(candidate_features)

    _plot_lasso_sweep(lasso_sweep, output_dir)
    _plot_feature_sources(pool_df, candidate_features, output_dir)

    # 2) Evaluate initial pool and prune variables under the actual profit function.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    final_features, final_eval, pruning_history = _greedy_backward_pruning(
        X_train,
        y_train,
        candidate_features,
        cv,
        min_features=1,
    )

    # 3) Build comparison tables/plots.
    initial_eval = _evaluate_feature_set(
        X_train,
        y_train,
        candidate_features,
        cv,
        stage="before_pruning",
    )
    after_eval = _evaluate_feature_set(
        X_train,
        y_train,
        final_features,
        cv,
        stage="after_pruning",
    )

    comparison_rows = initial_eval.all_rows + after_eval.all_rows
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = os.path.join(output_dir, "combined_model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"      Saved model comparison: {comparison_path}")

    history_path = os.path.join(output_dir, "combined_pruning_history.csv")
    pruning_history.to_csv(history_path, index=False)
    print(f"      Saved pruning history: {history_path}")

    _plot_model_comparison(comparison_rows, output_dir)
    _plot_pruning_history(pruning_history, output_dir)
    _plot_topk_profit_curve(final_eval, output_dir)
    _plot_weight_sweep(final_eval, output_dir)
    _plot_probability_distribution(_as_numpy_y(y_train), final_eval.best_probs, output_dir)

    # 4) Refit on full training data and predict test clients.
    print("[5/5] Final full-data training and test prediction")
    test_probs = _predict_test_probs(
        X_train,
        y_train,
        X_test,
        final_features,
        weight_xgb=final_eval.best_weight_xgb,
    )

    n_select = min(final_eval.best_k, MAX_CLIENTS)
    top_indices = np.argsort(test_probs)[::-1][:n_select]
    selected_clients = [int(i) + 1 for i in top_indices]
    used_features = [
        _feature_to_submission_index(f, X_train.columns)
        for f in final_features
    ]

    summary = pd.DataFrame(
        [
            {
                "final_model": final_eval.best_model,
                "xgb_weight": final_eval.best_weight_xgb,
                "n_features": len(final_features),
                "features": ",".join(map(str, final_features)),
                "submission_feature_ids": ",".join(map(str, used_features)),
                "selected_clients": len(selected_clients),
                "cv_profit": final_eval.best_profit,
                "opt_k": final_eval.best_k,
            }
        ]
    )
    summary_path = os.path.join(output_dir, "combined_final_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"      Saved final summary: {summary_path}")

    print("\nFinal combined strategy summary")
    print("-" * 40)
    print(f"Model             : {final_eval.best_model}")
    print(f"XGBoost weight    : {final_eval.best_weight_xgb:.1f}")
    print(f"OOF profit        : {final_eval.best_profit:.1f} EUR")
    print(f"Selected clients  : {len(selected_clients)}")
    print(f"Used variables    : {len(used_features)}")
    print(f"Feature cost      : {len(used_features) * VARIABLE_COST} EUR")
    print(f"Features          : {final_features}")

    result = StrategyResult(
        strategy="combined",
        selected_clients=selected_clients,
        used_features=used_features,
        estimated_profit=final_eval.best_profit,
        opt_k=final_eval.best_k,
        model_label=final_eval.best_model,
        validation_scheme="5-fold OOF CV after candidate generation",
        extra={
            "feature_names": ",".join(final_features),
            "xgb_weight": float(final_eval.best_weight_xgb),
            "summary_csv": summary_path,
        },
    )
    summary_path_std = save_strategy_summary(result, output_dir, "combined_summary.csv")
    print(f"      Saved standard summary: {summary_path_std}")
    return result
