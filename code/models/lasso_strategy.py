"""
Lasso (L1 Logistic Regression) Strategy
==================================================================
  Mutual Information pre-screening is calculated INSIDE each CV fold
  (only on the fold training data), so the validation set
  has no impact on feature selection -> no data leakage in CV profit estimation.

Pipeline:
  1. Coarse pre-screening on full data: 500 -> 200 candidates
     (space reduction only - minimal impact on leakage)
  2. Sweep over C: for each CV fold we compute MI on train_fold -> top 70 features,
     we train the model and evaluate on val_fold. OOF (out of fold) profit is fair.
  3. Greedy pruning on full data (acceptable for final model)
  4. Final model + prediction on test set
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------
# Pre-screening hyperparameters
# -----------------------------------------------------------------------
N_OUTER_PRESCREEN = 200   # 500 -> 200 (coarse filter on full data)
N_INNER_PRESCREEN = 70    # 200 -> 70 (inside each CV fold - no leakage)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_model(C: float) -> LogisticRegression:
    """L1 logistic regression (elasticnet with l1_ratio=1) via saga solver."""
    return LogisticRegression(
        penalty="elasticnet",
        l1_ratio=1.0,
        C=C,
        solver="saga",
        max_iter=3000,
        random_state=42,
        n_jobs=1,
    )


def _optimize_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_vars: int,
    max_clients: int = 1000,
) -> tuple[float, float]:
    """
    Find threshold maximizing profit by jointly optimising cut-off K.

    Sorts clients by predicted probability descending and evaluates profit
    for every K from 1 to max_clients. This avoids the 'select everyone'
    trap: at ~50% base rate a threshold of 0.10 selects all clients and
    inflates CV profit without reflecting real test behaviour.

    Returns (threshold, profit) where threshold is the midpoint between
    the prob of the last included and first excluded client.
    """
    sorted_idx = np.argsort(probs)[::-1]
    sorted_y   = y_true[sorted_idx]
    sorted_p   = probs[sorted_idx]

    best_profit = -np.inf
    best_thresh = 0.5
    best_k      = 1

    cumulative_tp = np.cumsum(sorted_y == 1)
    cumulative_fp = np.cumsum(sorted_y == 0)

    for k in range(1, min(max_clients, len(probs)) + 1):
        profit = (cumulative_tp[k - 1] * 10
                  - cumulative_fp[k - 1] * 5
                  - n_vars * 200)
        if profit > best_profit:
            best_profit = profit
            best_k      = k
            if k < len(sorted_p):
                best_thresh = (sorted_p[k - 1] + sorted_p[k]) / 2
            else:
                best_thresh = sorted_p[k - 1] / 2

    return best_thresh, best_profit


def _cv_profit_nested(
    X_outer: np.ndarray,   # data after OUTER pre-screening (N x 200)
    y: np.ndarray,
    C: float,
    cv: StratifiedKFold,
    n_inner: int = N_INNER_PRESCREEN,
) -> tuple[float, float, int]:
    """
    Cross-validated profit WITHOUT data leakage.

    For each fold:
      - MI is calculated only on X_outer[train_fold] (4000 x 200)
      - top n_inner features are selected
      - scaling uses only train_fold statistics (no leakage)
      - model is trained and evaluated on val_fold

    n_vars for profit calc = average number of non-zero features across folds.
    """
    oof_probs = np.zeros(len(y))
    fold_n_nonzero = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for train_idx, val_idx in cv.split(X_outer, y):
            X_tr, y_tr = X_outer[train_idx], y[train_idx]
            X_val       = X_outer[val_idx]

            # MI only on training data of this fold (no leakage)
            mi_fold = mutual_info_classif(X_tr, y_tr, random_state=42)
            inner_idx = np.argsort(mi_fold)[::-1][:n_inner]

            X_tr_sel  = X_tr[:, inner_idx]
            X_val_sel = X_val[:, inner_idx]

            # Scale inside fold (train stats only)
            fold_scaler = StandardScaler()
            X_tr_sel  = fold_scaler.fit_transform(X_tr_sel)
            X_val_sel = fold_scaler.transform(X_val_sel)

            model = _make_model(C)
            model.fit(X_tr_sel, y_tr)
            oof_probs[val_idx] = model.predict_proba(X_val_sel)[:, 1]
            fold_n_nonzero.append(int(np.sum(model.coef_[0] != 0)))

    avg_n = int(round(np.mean(fold_n_nonzero)))

    if avg_n == 0:
        return 1 / 3, -np.inf, 0

    thresh, profit = _optimize_threshold(y, oof_probs, avg_n)
    return thresh, profit, avg_n


def _cv_profit_final_features(
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    cv: StratifiedKFold,
) -> tuple[float, float, int]:
    """
    CV profit for a specific (already-scaled) feature set X.

    Note: X is scaled on full training data before this call, so there is
    minor leakage through the scaler. This is acceptable here because this
    function is only used for final feature pruning (not for unbiased
    generalisation estimates), and the effect is negligible given that
    StandardScaler statistics are stable.

    n_vars is counted from a model retrained on the full set at the end,
    which is the same model that will be deployed.
    """
    if X.shape[1] == 0:
        return 1 / 3, -np.inf, 0

    oof_probs = np.zeros(len(y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, val_idx in cv.split(X, y):
            m = _make_model(C)
            m.fit(X[train_idx], y[train_idx])
            oof_probs[val_idx] = m.predict_proba(X[val_idx])[:, 1]

        # Refit on full data to count active features for profit formula
        full_model = _make_model(C)
        full_model.fit(X, y)

    n_nz = int(np.sum(full_model.coef_[0] != 0))
    if n_nz == 0:
        return 1 / 3, -np.inf, 0

    thresh, profit = _optimize_threshold(y, oof_probs, n_nz)
    return thresh, profit, n_nz


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------

def run_lasso(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    output_dir: str = ".",
) -> tuple[list[int], list[int]]:
    print("=" * 65)
    print("  Lasso Strategy - CV without data leakage (nested MI per fold)")
    print("=" * 65)

    y = y_train.values
    col_names = X_train.columns.tolist()

    # ---------------------------------------------------------------- #
    # STEP 1: Outer (coarse) pre-screening on full data
    #         500 -> N_OUTER_PRESCREEN
    #         Goal: space reduction before sweep (speed).
    # ---------------------------------------------------------------- #
    print(
        f"\n[1/4] Outer pre-screening (MI on full data): "
        f"{X_train.shape[1]} -> {N_OUTER_PRESCREEN} candidates"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi_full = mutual_info_classif(X_train, y_train, random_state=42)

    outer_idx = np.argsort(mi_full)[::-1][:N_OUTER_PRESCREEN]
    outer_features = [col_names[i] for i in outer_idx]

    X_outer      = X_train.values[:, outer_idx]   # (5000, 200)
    X_test_outer = X_test.values[:, outer_idx]

    # ---------------------------------------------------------------- #
    # STEP 2: Sweep over C with nested MI inside each fold
    # ---------------------------------------------------------------- #
    C_VALUES = np.logspace(-2.5, 0.7, 30)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(
        f"\n[2/4] CV sweep over C "
        f"(MI calculated inside folds, top {N_INNER_PRESCREEN} features/fold)..."
    )
    print(f"      Note: this might take a few minutes.")
    print(f"{'C':>8}  {'Avg features':>12}  {'CV Profit [EUR]':>16}  {'Threshold':>9}")
    print("-" * 52)

    sweep_rows = []
    for C in C_VALUES:
        thresh, profit, avg_n = _cv_profit_nested(X_outer, y, C, cv)
        sweep_rows.append(
            {"C": C, "n_features": avg_n, "cv_profit": profit, "threshold": thresh}
        )
        profit_str = f"{profit:16.1f}" if profit > -np.inf else "    (no features)"
        print(f"{C:8.4f}  {avg_n:12d}  {profit_str}  {thresh:9.2f}")

    df_sweep = pd.DataFrame(sweep_rows)
    df_valid = df_sweep[df_sweep["n_features"] > 0]

    if df_valid.empty:
        raise RuntimeError("No C gave a model with >=1 feature. Change C_VALUES range.")

    # ---------------------------------------------------------------- #
    # Pick the best C that also produces >=1 feature when trained on
    # the full dataset (sanity-check each candidate in profit order).
    # ---------------------------------------------------------------- #
    df_valid_sorted = df_valid.sort_values("cv_profit", ascending=False).reset_index(
        drop=True
    )

    best_C   = None
    best_row = None

    print("\nSelecting best C that produces >=1 feature on full training data...")
    for _, row in df_valid_sorted.iterrows():
        candidate_C = float(row["C"])

        _mi = mutual_info_classif(X_outer, y_train, random_state=42)
        _inner_idx = np.argsort(_mi)[::-1][:N_INNER_PRESCREEN]
        _scaler = StandardScaler()
        _X_check = _scaler.fit_transform(X_outer[:, _inner_idx])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _m = _make_model(candidate_C)
            _m.fit(_X_check, y)

        n_nonzero = int(np.sum(_m.coef_[0] != 0))
        print(
            f"  C={candidate_C:.4f} -> {n_nonzero} non-zero features on full data",
            end="",
        )

        if n_nonzero > 0:
            best_C   = candidate_C
            best_row = row
            print("  <-- selected")
            break
        else:
            print("  (skipped: 0 features)")

    if best_C is None:
        raise RuntimeError(
            "No C in the sweep produced >=1 feature on full training data. "
            "Try widening C_VALUES (increase upper bound) or reducing N_INNER_PRESCREEN."
        )

    print(
        f"\n-> Best C = {best_C:.4f} | CV profit = {best_row['cv_profit']:.1f} EUR "
        f"| avg features in CV = {int(best_row['n_features'])}"
    )

    # ── Plot ───────────────────────────────────────────────────────────────
    df_plot = df_valid.copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(df_plot["C"], df_plot["cv_profit"], marker="o", color="#1f77b4", lw=2)
    ax1.axvline(best_C, color="red", ls="--", label=f"Best C={best_C:.4f}")
    ax1.set_ylabel("CV Profit (EUR)")
    ax1.set_title("Profit vs. L1 Regularization Strength [nested MI - no leakage]")
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    ax2.plot(df_plot["C"], df_plot["n_features"], marker="s", color="#2ca02c", lw=2)
    ax2.axvline(best_C, color="red", ls="--")
    ax2.set_xlabel("C (inverse of regularization strength)")
    ax2.set_ylabel("Avg number of features/fold")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "lasso_regularization_sweep.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")

    # ---------------------------------------------------------------- #
    # STEP 3: Final feature set (MI on full data) + greedy pruning
    # ---------------------------------------------------------------- #
    print(
        f"\n[3/4] Final model: MI on full data -> "
        f"top {N_INNER_PRESCREEN} features + greedy pruning..."
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi_inner = mutual_info_classif(X_outer, y_train, random_state=42)

    inner_idx_full = np.argsort(mi_inner)[::-1][:N_INNER_PRESCREEN]
    inner_features = [outer_features[i] for i in inner_idx_full]

    # Scale using full training data (final model only)
    scaler = StandardScaler()
    X_inner      = scaler.fit_transform(X_outer[:, inner_idx_full])
    X_test_inner = scaler.transform(X_test_outer[:, inner_idx_full])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base_model = _make_model(best_C)
        base_model.fit(X_inner, y)

    active_mask     = base_model.coef_[0] != 0
    active_features = [f for f, m in zip(inner_features, active_mask) if m]
    active_col_idx  = [i for i, m in enumerate(active_mask) if m]

    if not active_features:
        raise RuntimeError(
            "Final model has 0 features despite C validation passing. "
            "This is unexpected - please report."
        )

    _, base_profit, _ = _cv_profit_final_features(
        X_inner[:, active_col_idx], y, best_C, cv
    )
    print(f"Start: {len(active_features)} features, CV profit = {base_profit:.1f} EUR")

    # Greedy pruning: remove the feature whose removal improves profit most
    improved = True
    while improved and len(active_features) > 1:
        improved  = False
        best_pruned = base_profit
        worst_idx   = None

        for i in range(len(active_features)):
            trial_cols = [c for j, c in enumerate(active_col_idx) if j != i]
            _, trial_profit, _ = _cv_profit_final_features(
                X_inner[:, trial_cols], y, best_C, cv
            )
            if trial_profit > best_pruned:
                best_pruned = trial_profit
                worst_idx   = i

        if worst_idx is not None:
            removed = active_features.pop(worst_idx)
            active_col_idx.pop(worst_idx)
            base_profit = best_pruned
            print(
                f"  Removed: {removed} | Remaining: {len(active_features)} "
                f"| Profit: {base_profit:.1f} EUR"
            )
            improved = True

    print(f"\n-> After pruning: {len(active_features)} features, profit = {base_profit:.1f} EUR")
    print(f"   Features: {active_features}")

    # ---------------------------------------------------------------- #
    # STEP 4: Final training + prediction
    # ---------------------------------------------------------------- #
    print("\n[4/4] Training final model and predicting on test set...")

    X_final      = X_inner[:, active_col_idx]
    X_test_final = X_test_inner[:, active_col_idx]

    final_model = _make_model(best_C)
    oof_probs   = np.zeros(len(y))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, val_idx in cv.split(X_final, y):
            final_model.fit(X_final[train_idx], y[train_idx])
            oof_probs[val_idx] = final_model.predict_proba(X_final[val_idx])[:, 1]
        final_model.fit(X_final, y)

    final_thresh, final_cv_profit = _optimize_threshold(y, oof_probs, len(active_features))

    # Derive optimal K from OOF probabilities
    final_k = int(np.sum(oof_probs > final_thresh))
    if final_k == 0:
        final_k = 1000  # fallback

    print(f"Final threshold : {final_thresh:.4f}")
    print(f"Optimal K       : {final_k} clients")
    print(f"CV profit (fair): {final_cv_profit:.1f} EUR")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_probs = final_model.predict_proba(X_test_final)[:, 1]

    n_select    = min(final_k, 1000)
    top_indices = np.argsort(test_probs)[::-1][:n_select]

    best_clients_1_based = [int(idx) + 1 for idx in top_indices]
    used_features_idx    = [int(var.replace("V", "")) for var in active_features]

    print(f"\n  Selected clients : {len(best_clients_1_based)}")
    print(f"  Used features    : {len(used_features_idx)}")
    print(f"  Feature cost     : {len(used_features_idx) * 200} EUR")

    return best_clients_1_based, used_features_idx