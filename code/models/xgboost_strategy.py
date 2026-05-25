import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from core.metrics import calculate_profit


def run_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame):
    print("Initializing XGBoost strategy...")

    xgb_params = {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "scale_pos_weight": 2,
        "random_state": 42,
        "n_jobs": -1,
    }

    model_full = xgb.XGBClassifier(**xgb_params)
    model_full.fit(X_train, y_train)

    importances = pd.Series(model_full.feature_importances_, index=X_train.columns)
    importances = importances[importances > 0].sort_values(ascending=False)

    print("\n--- Feature Optimization Ranking (5-Fold CV) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_profit = -np.inf
    best_num_features = 0
    profit_ranking = []

    max_features_to_test = min(30, len(importances))

    for k in range(1, max_features_to_test + 1):
        current_features = importances.index[:k].tolist()
        oof_preds = np.zeros(len(X_train))

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr = X_train.iloc[train_idx][current_features]
            y_tr = y_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx][current_features]

            fold_model = xgb.XGBClassifier(**xgb_params)
            fold_model.fit(X_tr, y_tr)

            probs = fold_model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = (probs > 0.333).astype(int)

        current_profit = calculate_profit(y_train.values, oof_preds, num_variables=k)
        profit_ranking.append((k, current_profit))

        print(f"Features: {k:2d} | Estimated Training Profit (CV): {current_profit:7.1f} EUR")

        if current_profit > best_profit:
            best_profit = current_profit
            best_num_features = k

    features, profits = zip(*profit_ranking)

    plt.figure(figsize=(10, 6))
    plt.plot(features, profits, marker='o', linestyle='-', color='#1f77b4', linewidth=2)
    plt.title('Profit Optimization Curve vs Number of Features')
    plt.xlabel('Number of Selected Features (Top K)')
    plt.ylabel('Estimated Validation Profit (EUR)')
    plt.axvline(x=best_num_features, color='red', linestyle='--', label=f'Optimum ({best_num_features} features)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_optimization_profit.png', dpi=300)
    plt.close()
    print("Saved plot: feature_optimization_profit.png")

    top_features = importances.index[:best_num_features].tolist()
    print("-" * 46)
    print(f"Optimal number of features: {best_num_features} (Expected Profit: {best_profit} EUR)")
    print(f"Selected features: {top_features}\n")

    model_reduced = xgb.XGBClassifier(**xgb_params)
    model_reduced.fit(X_train[top_features], y_train)

    test_probs = model_reduced.predict_proba(X_test[top_features])[:, 1]
    test_predictions = (test_probs > 0.333).astype(int)

    selected_indices = np.where(test_predictions == 1)[0]
    best_probs_series = pd.Series(test_probs[selected_indices], index=selected_indices)
    best_clients_indices = best_probs_series.nlargest(1000).index.tolist()

    best_clients_1_based = [idx + 1 for idx in best_clients_indices]
    used_features_idx = [int(var.replace('V', '')) for var in top_features]

    return best_clients_1_based, used_features_idx