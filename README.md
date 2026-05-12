# AML Project 2: Cost-Sensitive Predictive Modeling

## Authors
* [**Norbert Frydrysiak**](https://github.com/fantasy2fry)
* [**Michał Kukla**](https://github.com/mickuk)
* [**Piotr Bartosiewicz**](https://github.com/PiotrDS)

## 📌 Project Overview
This repository contains the solution for Project 2 of the Advanced Machine Learning course. 

# TODO: Finish this chapter

## 🗂 Repository Structure
The project is structured into two main directories: `code/` and `report/`, adhering to the submission guidelines.

# TODO: Finish this chapter

* `code/data/`: 
* `code/src/`: Core Python modules.

* `code/notebooks/`: Contains `demo.ipynb`, a Jupyter notebook for newcomers who do not know the project yet and want a guided way to run the pipeline on new data.
* `code/main.py`: Main experiment runner (full experiment grid).

## 🚀 How to Run the Code

### 1. Prerequisites
Ensure you have Python 3.12 installed. It is recommended to use a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```


### 2. Run all experiments

From project root, run:

```bash
python3 code/main.py
```

This runs the full grid defined in `code/main.py` (datasets, schemes, missing rates, seeds, and methods) and uses process-level parallelization by default.

> Note: the current script does not expose CLI flags for partial runs. To change the scope, edit the `DEFAULT_*` constants in `code/main.py`.

### 3. Notebook quick start for new users

If someone is new to this project and wants to quickly try our approach on new data, start with:

- `code/notebooks/demo.ipynb`

The notebook is a guided entry point that explains the workflow step by step and is the easiest way to run and adapt the pipeline interactively.
