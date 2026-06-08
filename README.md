# AML Project 2: Cost-Sensitive Predictive Modeling

## Authors
* [**Norbert Frydrysiak**](https://github.com/fantasy2fry)
* [**Michał Kukla**](https://github.com/mickuk)
* [**Piotr Bartosiewicz**](https://github.com/PiotrDS)

## 📌 Project Overview
This repository contains the solution for Project 2 of the Advanced Machine Learning course. The objective of this project is to build a cost-sensitive predictive model to maximize expected profit by selecting optimal features and a subset of clients for a marketing campaign. We implemented multiple modeling strategies (XGBoost, Lasso, Forward Selection) and combined them to determine the strongest approach.

## 🗂 Repository Structure
The project is structured into several main directories:

* `code/`: Contains the source code of the project.
  * `core/`: Core modules for data loading, metrics, plotting, reporting.
  * `models/`: Implementations of different strategies (Forward Selection, Lasso, XGBoost, and Combined Strategy).
  * `main.py`: Main script to run the models.
* `data/`: Contains the training and testing datasets (`x_train.txt`, `y_train.txt`, `x_test.txt`).
* `submission/`: Output directory containing results, logs, plots, summary CSVs, and selected variables/observations.
* `run_all_models.sh`: Shell script to execute all strategies and generate a leaderboard.

## 🚀 How to Run the Code

### 1. Prerequisites
Ensure you have Python 3.12 installed. The project uses `uv` for dependency management (recommended), but you can also use traditional `pip`.

```bash
# Using uv (Recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Or using plain pip
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run all experiments

To evaluate all models and create a leaderboard, run the utility script from the project root:

```bash
./run_all_models.sh
```

but you need to have x_test.txt, x_train.txt and y_train.txt in the `data/` folder.

This script will ask for student IDs which are used to generate the final submission files, and then it runs the full experiment grid using `uv run`.

### 3. Running a specific model

If you prefer to run a single specific model, you can do so directly using the `code/main.py` script:

```bash
uv run python code/main.py --model combined --student_ids "ID1_ID2_ID3"
```

Available models are: `xgb`, `lasso`, `forward`, `combined`, `all`. The results, including models' logs, plots, and summary CSVs, are saved into the `submission/` folder.
