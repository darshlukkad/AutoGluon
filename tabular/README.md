# AutoGluon Tabular — Knot Theory Example: Code Walkthrough

This document explains what each line of your notebook/script does. It’s written for a Google Colab or Jupyter environment where shell commands (lines starting with `!`) are supported.

---

## TL;DR

1. Install/upgrade dependencies and AutoGluon.
2. Load a public train/test dataset from GitHub.
3. Inspect the target column `signature`.
4. Train an AutoGluon **TabularPredictor** on the training data.
5. Predict on the test set.
6. Evaluate the model and inspect the leaderboard.

---

## 1) Install/Upgrade Dependencies

```bash
!python -m pip install --upgrade pip
!python -m pip install autogluon
```

- Ensures `pip` is up to date and installs **AutoGluon** (the full meta-learner for tabular ML).
- In Colab, `!` runs shell commands. If you’re on a local machine, you can run these in a terminal instead.

---

## 2) Imports

```python
from autogluon.tabular import TabularDataset, TabularPredictor
```

- `TabularDataset` wraps CSVs (or dataframes) with lightweight conveniences for AutoGluon.
- `TabularPredictor` is the high-level AutoGluon interface for training, inference, and evaluation on tabular data.

---

## 3) Load the Training Data

```python
data_url = 'https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/'
train_data = TabularDataset(f'{data_url}train.csv')
train_data.head()
```

- Downloads `train.csv` directly from GitHub and loads it as an AutoGluon `TabularDataset`.
- `head()` shows the first few rows so you can quickly verify columns and dtypes.

> **Note:** Remote CSVs work out of the box; AutoGluon handles the download.

---

## 4) Declare the Target Column and Inspect It

```python
label = 'signature'
train_data[label].describe()
```

- Sets the column to predict: **`signature`**.
- `describe()` prints basic stats.  
  AutoGluon will **infer the problem type** (classification vs regression) from the target’s dtype and values.

> If `signature` is numeric and continuous → regression; if it’s categorical (string / discrete classes) → classification.  
> You can override with `TabularPredictor(problem_type='regression' or 'multiclass' or 'binary')` if needed.

---

## 5) Train the Model

```python
predictor = TabularPredictor(label=label).fit(train_data)
```

- Creates and **fits** a tabular predictor:
  - Learns the mapping from features → `signature`.
  - Automatically does preprocessing, model selection, and ensembling.
  - Uses sensible defaults for presets, validation strategy, and evaluation metric (based on inferred problem type).

**Common optional controls:**
```python
predictor = TabularPredictor(
    label=label,
    eval_metric='roc_auc'  # or 'accuracy', 'r2', 'mean_squared_error', etc., depending on problem
).fit(
    train_data,
    presets='medium_quality',   # speed/accuracy trade-off
    time_limit=3600             # seconds, to cap training time
)
```

---

## 6) Load Test Data & Predict

```python
test_data = TabularDataset(f'{data_url}test.csv')

y_pred = predictor.predict(test_data.drop(columns=[label]))
y_pred.head()
```

- Loads `test.csv`.
- Drops the label column before calling `.predict(...)` (you only pass features at inference time).
- Returns predictions:
  - **Classification:** class labels (use `.predict_proba(...)` for probabilities).
  - **Regression:** numeric predictions.

> If `test.csv` doesn’t contain the label, skip the `drop(columns=[label])` and just pass `test_data` to `predict`.

---

## 7) Evaluate on Test Data

```python
predictor.evaluate(test_data, silent=True)
```

- Computes a metric appropriate to the inferred problem type **using the label column present in `test_data`**.
- `silent=True` suppresses extra logs; the return value is the score (e.g., accuracy for classification, R² for regression).

> Ensure `test.csv` includes the true `signature` column; otherwise evaluation will fail.

---

## 8) Inspect the Leaderboard

```python
predictor.leaderboard(test_data)
```

- Displays a model leaderboard as a dataframe, including:
  - Model names (e.g., LightGBM, CatBoost, neural net, stackers).
  - Validation scores and (if test data is provided) test scores.
  - Fit times, prediction times, and other diagnostics.

**Tip:** Save it for later analysis:
```python
lb = predictor.leaderboard(test_data, silent=True)
lb.to_csv('leaderboard.csv', index=False)
```

---

## Practical Tips & Extensions

- **Problem Type Control:** If AutoGluon guesses wrong, set `problem_type` explicitly in `TabularPredictor(...)`.
- **Metrics:** Pick a metric that matches your goals (e.g., `roc_auc` for imbalanced binary classification).
- **Presets & Time:** `best_quality` often yields stronger results but takes longer; consider setting `time_limit`.
- **Feature Importance:** After training, run `predictor.feature_importance(train_data)` to see which features matter.
- **Reproducibility:** Use fixed seeds where relevant and pin package versions for deterministic behavior.
- **Data Checks:** Confirm there’s no leakage (e.g., identifiers or post-outcome features) and that train/test schemas match.

---

**End of walkthrough.** Happy modeling! 🎓
