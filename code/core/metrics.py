import numpy as np


def calculate_profit(
    y_true: np.ndarray, y_pred: np.ndarray, num_variables: int
) -> float:
    """Compute profit for a binary contact decision.
    Note: the project evaluation typically caps contacted clients at 1000; callers
    should enforce that cap before calling this helper.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    score = (tp * 10) - (fp * 5) - (num_variables * 200)

    return float(score)
