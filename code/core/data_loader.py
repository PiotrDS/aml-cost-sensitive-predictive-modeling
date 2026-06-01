from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def load_data(data_dir: str | os.PathLike[str]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load training variables, training labels and test variables.

    Args:
        data_dir: Directory containing ``x_train.txt``, ``y_train.txt`` and
            ``x_test.txt``.

    Returns:
        A tuple ``(X_train, y_train, X_test)`` where the feature matrices are
        pandas DataFrames and the label vector is a pandas Series.
    """
    data_path = Path(data_dir)
    x_train_path = data_path / "x_train.txt"
    y_train_path = data_path / "y_train.txt"
    x_test_path = data_path / "x_test.txt"

    X_train = pd.read_csv(x_train_path, sep=r"\s+")
    X_test = pd.read_csv(x_test_path, sep=r"\s+")
    y_train = pd.read_csv(y_train_path, sep=r"\s+")["y"]

    return X_train, y_train, X_test
