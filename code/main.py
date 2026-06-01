import argparse
import os

from core.data_loader import load_data
from models.lasso_strategy import run_lasso
from models.xgboost_strategy import run_xgboost
from models.forward_strategy import run_forward
from models.combined_strategy import run_combined


def main():
    parser = argparse.ArgumentParser(description="AML Project 2: Cost-Sensitive Model")
    parser.add_argument("--model", choices=["xgb", "lasso", "forward", "combined"], default="combined")
    parser.add_argument(
        "--student_ids",
        type=str,
        default="123456_98765_98764",
        help="Student IDs in format ID1_ID2_ID3",
    )
    args = parser.parse_args()

    # ── Data ──────────────────────────────────────────────────────────────
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    X_train, y_train, X_test = load_data(data_dir)

    print(f"Data Loaded: X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"Class balance: {y_train.mean():.3f} positive rate\n")

    # ── Output directory:  submission/<model>/ ────────────────────────────
    model_name = {
        "xgb": "xgboost",
        "lasso": "lasso",
        "forward": "forward",
        "combined": "combined",
    }[args.model]
    output_dir = os.path.join(os.path.dirname(__file__), "..", "submission", model_name)
    os.makedirs(output_dir, exist_ok=True)

    # ── Run chosen strategy ───────────────────────────────────────────────
    if args.model == "xgb":
        selected_clients, used_features = run_xgboost(
            X_train, y_train, X_test, output_dir=output_dir
        )
    elif args.model == "lasso":
        selected_clients, used_features = run_lasso(
            X_train, y_train, X_test, output_dir=output_dir
        )
    elif args.model == "forward":
        selected_clients, used_features = run_forward(
            X_train, y_train, X_test, output_dir=output_dir
        )
    elif args.model == "combined":
        selected_clients, used_features = run_combined(
            X_train, y_train, X_test, output_dir=output_dir
        )

    # ── Save submission files ─────────────────────────────────────────────
    student_ids = args.student_ids

    obs_path = os.path.join(output_dir, f"{student_ids}_obs.txt")
    with open(obs_path, "w") as f:
        for client_id in selected_clients[:1000]:
            f.write(f"{client_id}\n")

    vars_path = os.path.join(output_dir, f"{student_ids}_vars.txt")
    with open(vars_path, "w") as f:
        for var in used_features:
            f.write(f"{var}\n")

    print(f"\nSaved: {obs_path}  ({len(selected_clients[:1000])} clients)")
    print(f"Saved: {vars_path}  ({len(used_features)} features)")


if __name__ == "__main__":
    main()
