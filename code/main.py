from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path
from typing import Callable

from core.data_loader import load_data
from core.reporting import (
    StrategyResult,
    coerce_strategy_result,
    plot_leaderboard,
    save_leaderboard_csv,
    save_strategy_summary,
    save_submission_files,
    tee_output,
)
from models.combined_strategy import run_combined
from models.forward_strategy import run_forward
from models.lasso_strategy import run_lasso
from models.xgboost_strategy import run_xgboost

StrategyRunner = Callable[..., StrategyResult]

MODEL_REGISTRY: dict[str, tuple[str, StrategyRunner]] = {
    "xgb": ("xgboost", run_xgboost),
    "lasso": ("lasso", run_lasso),
    "forward": ("forward", run_forward),
    "combined": ("combined", run_combined),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a single model or the full run."""
    parser = argparse.ArgumentParser(description="AML Project 2: cost-sensitive predictive modeling")
    parser.add_argument(
        "--model",
        choices=[*MODEL_REGISTRY.keys(), "all"],
        default="combined",
        help="Modeling strategy to run. Use 'all' to run all strategies and create a leaderboard.",
    )
    parser.add_argument(
        "--student_ids",
        type=str,
        default="123456_98765_98764",
        help="Student IDs in format ID1_ID2_ID3.",
    )
    return parser.parse_args()


def project_root() -> Path:
    """Return the repository root assuming this file is located in ``code/``."""
    return Path(__file__).resolve().parent.parent


def model_keys_to_run(model_argument: str) -> list[str]:
    """Return model keys requested by the command-line argument."""
    if model_argument == "all":
        return list(MODEL_REGISTRY.keys())
    return [model_argument]


def run_one_strategy(
    model_key: str,
    student_ids: str,
    data_dir: Path,
    submission_root: Path,
) -> StrategyResult:
    """Run one strategy, capture its log and write submission artifacts."""
    model_dir_name, runner = MODEL_REGISTRY[model_key]
    output_dir = submission_root / model_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{model_dir_name}_run.log"

    with tee_output(log_path):
        try:
            print("=" * 72)
            print(f"Running strategy: {model_dir_name}")
            print("=" * 72)

            X_train, y_train, X_test = load_data(data_dir)
            print(f"Data loaded: X_train {X_train.shape}, X_test {X_test.shape}")
            print(f"Class balance: {y_train.mean():.3f} positive rate\n")

            raw_output = runner(X_train, y_train, X_test, output_dir=str(output_dir))
            result = coerce_strategy_result(raw_output, strategy=model_dir_name)
            result.strategy = model_dir_name
            result.log_path = str(log_path)

            save_submission_files(result, output_dir, student_ids)
            summary_path = save_strategy_summary(result, output_dir)

            print(f"\nSaved: {result.obs_path}  ({len(result.selected_clients[:1000])} clients)")
            print(f"Saved: {result.vars_path}  ({len(result.used_features)} features)")
            print(f"Saved summary: {summary_path}")
            print(f"Saved run log: {log_path}")
            return result

        except Exception:
            print("\nStrategy failed. Full traceback:")
            traceback.print_exc()
            raise


def save_full_leaderboard(results: list[StrategyResult], submission_root: Path) -> None:
    """Save CSV and PNG leaderboard for all strategies executed in this run."""
    leaderboard_dir = submission_root / "leaderboard"
    leaderboard, csv_path = save_leaderboard_csv(results, leaderboard_dir)
    plot_path = plot_leaderboard(leaderboard, leaderboard_dir)

    print("\nStrategy leaderboard")
    print("-" * 72)
    if leaderboard.empty:
        print("No leaderboard rows were created.")
        return

    display_columns = [
        "rank",
        "strategy",
        "estimated_profit",
        "validation_scheme",
        "selected_clients",
        "n_features",
        "feature_cost_eur",
    ]
    print(leaderboard[display_columns].to_string(index=False))
    print(f"\nSaved leaderboard CSV: {csv_path}")
    if plot_path is not None:
        print(f"Saved leaderboard plot: {plot_path}")


def main() -> None:
    """Run the requested strategy or all strategies from the command line."""
    args = parse_args()
    root = project_root()
    data_dir = root / "data"
    submission_root = root / "submission"
    submission_root.mkdir(parents=True, exist_ok=True)

    results = []
    for model_key in model_keys_to_run(args.model):
        result = run_one_strategy(
            model_key=model_key,
            student_ids=args.student_ids,
            data_dir=data_dir,
            submission_root=submission_root,
        )
        results.append(result)

    if len(results) > 1:
        save_full_leaderboard(results, submission_root)


if __name__ == "__main__":
    main()
