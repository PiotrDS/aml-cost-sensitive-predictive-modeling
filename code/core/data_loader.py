import os

import pandas as pd


def load_data(data_dir: str):
    x_train_path = os.path.join(data_dir, "x_train.txt")
    y_train_path = os.path.join(data_dir, "y_train.txt")
    x_test_path = os.path.join(data_dir, "x_test.txt")

    X_train = pd.read_csv(x_train_path, sep=r"\s+")
    X_test = pd.read_csv(x_test_path, sep=r"\s+")

    y_train_df = pd.read_csv(y_train_path, sep=r"\s+")
    y_train = y_train_df["y"]

    return X_train, y_train, X_test
