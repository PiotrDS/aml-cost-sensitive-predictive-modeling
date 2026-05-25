import argparse
import os

from core.data_loader import load_data
from models.xgboost_strategy import run_xgboost

# from models.lasso_strategy import run_lasso
# from models.stepwise_strategy import run_stepwise


def main():
    parser = argparse.ArgumentParser(description="AML Project 2: Cost-Sensitive Model")
    parser.add_argument("--model", choices=["xgb", "lasso", "stepwise"], default="xgb")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    X_train, y_train, X_test = load_data(data_dir)

    print(f"Data Loaded: X_train {X_train.shape}, X_test {X_test.shape}")

    if args.model == "xgb":
        selected_clients, used_features = run_xgboost(X_train, y_train, X_test)
    elif args.model == "lasso":
        pass  # selected_clients, used_features = run_lasso(X_train, y_train, X_test)
    elif args.model == "stepwise":
        pass  # selected_clients, used_features = run_stepwise(X_train, y_train, X_test)

    student_ids = "123456_98765_98764"

    obs_filename = f"{student_ids}_obs.txt"
    with open(obs_filename, "w") as f:
        for client_id in selected_clients[:1000]:
            f.write(f"{client_id}\n")

    vars_filename = f"{student_ids}_vars.txt"
    with open(vars_filename, "w") as f:
        for var in used_features:
            f.write(f"{var}\n")

    print(f"Saved: {obs_filename} and {vars_filename}")


if __name__ == "__main__":
    main()
