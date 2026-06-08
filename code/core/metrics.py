from __future__ import annotations

import numpy as np

TRUE_POSITIVE_REWARD = 10
FALSE_POSITIVE_COST = 5
VARIABLE_COST = 200
MAX_CLIENTS = 1000


def calculate_profit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_variables: int,
) -> float:
    """Compute the project profit for a binary contact decision vector.

    Args:
        y_true: Binary ground-truth labels where 1 means accepted offer.
        y_pred: Binary decision vector where 1 means selected for contact.
        num_variables: Number of variables used by the model.

    Returns:
        Profit calculated as ``10 * TP - 5 * FP - 200 * num_variables``.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return float(
        TRUE_POSITIVE_REWARD * tp
        - FALSE_POSITIVE_COST * fp
        - VARIABLE_COST * num_variables
    )


def top_k_profit_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    num_variables: int,
    max_clients: int = MAX_CLIENTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the profit curve obtained by selecting top-K predicted clients.

    Args:
        y_true: Binary validation labels.
        probabilities: Predicted conversion probabilities.
        num_variables: Number of variables used by the model.
        max_clients: Maximum number of customers allowed to contact.

    Returns:
        Two arrays: K values and the corresponding project profits.
    """
    y_true = np.asarray(y_true).ravel()
    probabilities = np.asarray(probabilities).ravel()
    limit = min(max_clients, len(y_true))

    order = np.argsort(probabilities)[::-1]
    y_sorted = y_true[order[:limit]]
    cumulative_tp = np.cumsum(y_sorted == 1)
    cumulative_fp = np.cumsum(y_sorted == 0)
    ks = np.arange(1, limit + 1)
    profits = (
        TRUE_POSITIVE_REWARD * cumulative_tp
        - FALSE_POSITIVE_COST * cumulative_fp
        - VARIABLE_COST * num_variables
    )
    return ks, profits.astype(float)


def optimize_top_k(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    num_variables: int,
    max_clients: int = MAX_CLIENTS,
) -> tuple[float, int]:
    """Find the best top-K cutoff under the project profit function."""
    ks, profits = top_k_profit_curve(y_true, probabilities, num_variables, max_clients)
    best_idx = int(np.argmax(profits))
    return float(profits[best_idx]), int(ks[best_idx])
