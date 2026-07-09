# Brake Friction Coefficient Prediction

A regression model that predicts the friction coefficient between a truck brake pad and disc, given 5 measured operating conditions from a brake dynamometer test rig.

## Problem

The friction coefficient between a brake pad and disc directly determines braking force, and is a key input for active safety systems like ABS and ESP. This project trains a data-driven regression model to predict it, given:

- `rotor_speed_rpm`
- `brake_pressure_bar`
- `normal_force_kN`
- `pad_temperature_C`
- `rotor_temperature_C`

## Data

`data_students.txt` — 227,978 measurements from a truck brake dynamometer, recorded during systematically varied stop-braking maneuvers. Each row is one measurement instant; the dataset is shuffled with no temporal structure.

## Model

A `RandomForestRegressor` (scikit-learn), tuned to balance accuracy against file size:

```python
RandomForestRegressor(
    n_estimators=50,
    min_samples_leaf=30,
    random_state=42,
)
```

**Why Random Forest:** compared against Linear Regression, K-Nearest Neighbors, Extra Trees, and Gradient Boosting via 5-fold cross-validation. Random Forest gave a strong, stable accuracy-to-file-size tradeoff. Tree-based models need no feature scaling, since splits compare raw feature values directly.

**Why these specific hyperparameters:** larger tree ensembles (e.g. `n_estimators=200` with default leaf size) reached R² > 0.998, but produced multi-gigabyte pickle files — impractical to distribute. `min_samples_leaf=30` keeps each tree's leaves representing genuine groups of examples rather than memorizing individual rows, keeping the model both smaller and better-generalizing.

## Results

5-fold cross-validated R² (honest, held-out estimate — not measured on training data):

```
R2 per fold: [0.96939 0.97207 0.9698  0.96928 0.96976]
Mean R2:     0.9701
Std dev:     0.0010
```

## Files

| File | Purpose |
|---|---|
| `train_random_forest.py` | Trains the final model on the full dataset and saves it |
| `evaluate_model.py` | Runs 5-fold cross-validation to produce an honest R² estimate |
| `try_prediction.py` | Load the saved model and predict on a custom input |
| `helpers.py` | Data loading / model save-load utilities |
| `YH_106.pkl` | The trained, serialized model |
| `data_students.txt` | Training data |

## Usage

```bash
pip install scikit-learn==1.7.0 numpy==2.3.1

python train_random_forest.py      # retrain from scratch
python evaluate_model.py           # reproduce the R2 estimate
python try_prediction.py           # try a custom prediction
```

## Context

Built as part of the "Applied Machine Learning in Engineering" course (TU Berlin, Summer Term 2026).
