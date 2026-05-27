import numpy as np


# it's good as a helper function, but we should always test on max 1000 clients
def calculate_profit(
    y_true: np.ndarray, y_pred: np.ndarray, num_variables: int
) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    score = (tp * 10) - (fp * 5) - (num_variables * 200)

    return float(score)
