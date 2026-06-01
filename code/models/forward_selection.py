"""Greedy feature selection optimized for the project profit function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score


@dataclass
class SubsetEvaluation:
    """Validation result for one candidate feature subset."""

    profit: float
    best_k: int
    threshold: float


class FeatureSelector:
    """Forward or backward greedy selector using a custom score function."""

    def __init__(
        self,
        model: BaseEstimator,
        score_function: Callable[[np.ndarray, np.ndarray, int], float],
        top_ns: Sequence[int] = (200, 300, 400),
        random_state: int | None = None,
        verbose: bool = True,
        features: list[int] | None = None,
        forward: bool = True,
        max_selected_features: int = 50,
    ) -> None:
        """Store selector configuration.

        Args:
            model: Estimator implementing ``fit`` and ``predict_proba``.
            score_function: Function receiving ``y_true``, ``y_pred`` and the
                number of features, returning the project profit.
            top_ns: Candidate values for the number of contacted customers.
            random_state: Optional random seed kept for API completeness.
            verbose: Whether to print the final selected subset.
            features: Initial candidate feature indexes.
            forward: If True, add features greedily; otherwise remove them.
            max_selected_features: Maximum number of features in forward mode.
        """
        self.model = model
        self.score_function = score_function
        self.top_ns = tuple(top_ns)
        self.random_state = random_state
        self.verbose = verbose
        self.features = list(features) if features is not None else None
        self.forward = forward
        self.max_selected_features = max_selected_features

    def _evaluate_subset(
        self,
        features: list[int],
        X_train: np.ndarray,
        X_valid: np.ndarray,
        y_train: np.ndarray,
        y_valid: np.ndarray,
    ) -> SubsetEvaluation:
        """Fit the model on a subset and return its best validation profit."""
        model = clone(self.model)
        model.fit(X_train[:, features], y_train)
        probabilities = model.predict_proba(X_valid[:, features])[:, 1]
        order = np.argsort(probabilities)[::-1]

        thresholds = (0.04, 0.05, 0.06, 0.08, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
        best = SubsetEvaluation(profit=-np.inf, best_k=0, threshold=0.5)

        for top_n in self.top_ns:
            selected = order[: min(top_n, len(order))]
            y_top = y_valid[selected]
            p_top = probabilities[selected]
            for threshold in thresholds:
                y_pred = (p_top > threshold).astype(int)
                profit = self.score_function(y_top, y_pred, len(features))
                if profit > best.profit:
                    best = SubsetEvaluation(float(profit), int(len(selected)), float(threshold))

        return best

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
    ) -> "FeatureSelector":
        """Select features by greedy profit improvement on the validation set."""
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train).ravel()
        X_valid = np.asarray(X_valid)
        y_valid = np.asarray(y_valid).ravel()

        candidates = self.features if self.features is not None else list(range(X_train.shape[1]))
        if self.forward:
            self._fit_forward(candidates, X_train, X_valid, y_train, y_valid)
        else:
            self._fit_backward(candidates, X_train, X_valid, y_train, y_valid)

        if self.verbose:
            print(
                "Features Selected: "
                f"{self.selected_features_}\n"
                f"Score Function: {self.best_score_}\n"
                f"Threshold: {self.best_threshold}\n"
                f"Candidates Number: {self.best_candidates}"
            )
        return self

    def _fit_forward(
        self,
        candidates: list[int],
        X_train: np.ndarray,
        X_valid: np.ndarray,
        y_train: np.ndarray,
        y_valid: np.ndarray,
    ) -> None:
        """Greedily add variables while the validation profit improves."""
        remaining = list(candidates)
        selected: list[int] = []
        best_score = -np.inf
        best_k = 0
        best_threshold = 0.5

        while remaining and len(selected) < self.max_selected_features:
            trial_results = []
            for feature in remaining:
                trial_features = selected + [feature]
                evaluation = self._evaluate_subset(trial_features, X_train, X_valid, y_train, y_valid)
                trial_results.append((feature, evaluation))

            best_feature, best_eval = max(trial_results, key=lambda item: item[1].profit)
            if best_eval.profit < best_score:
                break

            selected.append(best_feature)
            remaining.remove(best_feature)
            best_score = best_eval.profit
            best_k = best_eval.best_k
            best_threshold = best_eval.threshold

        self.selected_features_ = selected
        self.best_score_ = float(best_score)
        self.best_threshold = float(best_threshold)
        self.best_candidates = int(best_k)

    def _fit_backward(
        self,
        candidates: list[int],
        X_train: np.ndarray,
        X_valid: np.ndarray,
        y_train: np.ndarray,
        y_valid: np.ndarray,
    ) -> None:
        """Greedily remove variables while the validation profit improves."""
        selected = list(candidates)
        evaluation = self._evaluate_subset(selected, X_train, X_valid, y_train, y_valid)
        best_score = evaluation.profit
        best_k = evaluation.best_k
        best_threshold = evaluation.threshold

        improved = True
        while improved and len(selected) > 1:
            improved = False
            trial_results = []
            for feature in selected:
                trial_features = [item for item in selected if item != feature]
                trial_eval = self._evaluate_subset(trial_features, X_train, X_valid, y_train, y_valid)
                trial_results.append((feature, trial_features, trial_eval))

            _, best_trial_features, best_trial_eval = max(trial_results, key=lambda item: item[2].profit)
            if best_trial_eval.profit >= best_score:
                selected = best_trial_features
                best_score = best_trial_eval.profit
                best_k = best_trial_eval.best_k
                best_threshold = best_trial_eval.threshold
                improved = True

        self.selected_features_ = selected
        self.best_score_ = float(best_score)
        self.best_threshold = float(best_threshold)
        self.best_candidates = int(best_k)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the matrix restricted to the selected features."""
        return np.asarray(X)[:, self.selected_features_]

    def fit_transform(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
    ) -> np.ndarray:
        """Fit the selector and return transformed training data."""
        self.fit(X_train, y_train, X_valid, y_valid)
        return self.transform(X_train)


def score_function(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """Compute the project profit for a validation decision vector."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    return float(10 * tp - 5 * fp - 200 * n_features)


def single_feature_ranking(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    model: BaseEstimator,
    score_function: Callable[[np.ndarray, np.ndarray, int], float],
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rank variables by their single-feature validation profit."""
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()
    X_valid = np.asarray(X_valid)
    y_valid = np.asarray(y_valid).ravel()

    profits = []
    accuracies = []
    balanced_accuracies = []
    thresholds = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)

    for feature_idx in range(X_train.shape[1]):
        candidate_model = clone(model)
        candidate_model.fit(X_train[:, [feature_idx]], y_train)
        probabilities = candidate_model.predict_proba(X_valid[:, [feature_idx]])[:, 1]

        threshold_profits = []
        threshold_accuracies = []
        threshold_balanced_accuracies = []
        for threshold in thresholds:
            prediction = (probabilities > threshold).astype(int)
            threshold_profits.append(score_function(y_valid, prediction, 1))
            threshold_accuracies.append(accuracy_score(y_valid, prediction))
            threshold_balanced_accuracies.append(balanced_accuracy_score(y_valid, prediction))

        best_idx = int(np.argmax(threshold_profits))
        profits.append(threshold_profits[best_idx])
        accuracies.append(threshold_accuracies[best_idx])
        balanced_accuracies.append(threshold_balanced_accuracies[best_idx])

        if verbose:
            print(f"Feature {feature_idx}, best score: {threshold_profits[best_idx]}")

    ranking = np.argsort(profits)[::-1]
    return ranking, np.array(profits), np.array(accuracies), np.array(balanced_accuracies)


def forward_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    model: BaseEstimator = GradientBoostingClassifier(n_estimators=10),
) -> tuple[list[int], float, int]:
    """Run forward selection and return variables, threshold and selected K.

    The function first ranks all variables by single-feature performance, then
    runs greedy forward selection on the best 50 candidate variables.
    """
    ranking, _, _, _ = single_feature_ranking(
        X_train,
        y_train,
        X_valid,
        y_valid,
        model,
        score_function,
        verbose=False,
    )
    selector = FeatureSelector(
        model=model,
        score_function=score_function,
        top_ns=list(range(5, 1005, 5)),
        features=list(ranking[:50]),
        verbose=True,
    )
    selector.fit(X_train, y_train, X_valid, y_valid)
    return selector.selected_features_, selector.best_threshold, selector.best_candidates
