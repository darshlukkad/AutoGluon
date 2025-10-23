# AutoGluon Fraud Detection Pipeline — Code Walkthrough

This document explains, step by step, what each part of your notebook/script does. It’s written for a Google Colab or Jupyter environment where shell commands (lines starting with `!`) are supported.

---

## TL;DR (What this pipeline does)

1. Installs dependencies and downloads the competition data from Google Drive into `/content/data`.
2. Loads and merges the identity and transaction CSVs on `TransactionID`.
3. Stratified-samples **20%** of the training data (by the `isFraud` label) to speed up experimentation.
4. Trains an **AutoGluon Tabular** model (preset: `medium_quality`) using **ROC AUC** as the evaluation metric.
5. Aligns test columns to the training feature schema.
6. Predicts fraud probabilities for the test set and writes **`my_submission.csv`**.

---

## 1) Setup & Data Download

```bash
!pip install -U pip setuptools wheel
#!pip install -U autogluon autogluon.timeseries

# Install gdown and create the target directory
!pip -q install gdown
!mkdir -p /content/data

# Download the entire shared folder to /content/data
!gdown --folder --fuzzy "https://drive.google.com/drive/folders/1K6Zdl_rt8AH0XRE4ww_jbvUtP3Zm91Xa?usp=sharing" -O /content/data

# Quick check
!ls -lah /content/data
```

**What’s happening:**
- Upgrades core packaging tools.
- Installs **gdown** (a helper to download files/folders from Google Drive).
- Makes sure `/content/data` exists.
- Downloads the shared Drive folder into `/content/data`.
- Lists the downloaded files so you can verify they’re there.

> **Tip:** If you’re not in Colab, remove the leading `!` and run these shell commands in your system shell, or use Python equivalents (e.g., `subprocess.run`).

---

## 2) Imports & Basic Configuration

```python
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

directory = '/content/data/'        # where the CSVs live
label = 'isFraud'                   # target column to predict
eval_metric = 'roc_auc'             # competition metric: AUC
save_path = directory + 'AutoGluonModels/'  # model output dir
```

- `TabularPredictor` is AutoGluon’s high-level API for tabular data.
- Using **ROC AUC** (`roc_auc`) emphasizes ranking/pairwise discrimination—commonly used for fraud detection.

---

## 3) Load & Merge Train Tables

```python
train_identity = pd.read_csv(directory+'train_identity.csv')
train_transaction = pd.read_csv(directory+'train_transaction.csv')

train_data = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
```

- Loads two training tables and left-joins on **`TransactionID`** so every transaction keeps its record even if identity info is missing.
- The merged `train_data` contains features from both sources plus the label `isFraud`.

---

## 4) Speed-Up via Stratified 20% Sample

```python
# --- SAMPLE 20% OF TRAINING DATA (stratified by label) ---
train_data = (
    train_data.groupby(label, group_keys=False)
              .apply(lambda x: x.sample(frac=0.2, random_state=42))
              .reset_index(drop=True)
)
print(f"Train rows after 20% sample: {len(train_data):,}")
# -----------------------------------------------------------
```

- To iterate faster, you downsample to **20%** of rows **within each label class** (keeps class proportions).
- `random_state=42` gives reproducible samples.

> **Note:** For final training, remove this sampling or increase `frac`—more data often means better performance.

---

## 5) Train the AutoGluon Model

```python
predictor = TabularPredictor(label=label, eval_metric=eval_metric, path=save_path, verbosity=3).fit(
    train_data,
    presets='medium_quality',
    time_limit=3600
)

results = predictor.fit_summary()
```

- Creates a `TabularPredictor` that:
  - Knows the target column (`isFraud`).
  - Optimizes for **ROC AUC**.
  - Saves models/artefacts in `AutoGluonModels/`.
- `.fit(...)` starts training:
  - `presets='medium_quality'` balances speed and accuracy.
  - `time_limit=3600` caps training at **1 hour** (tune as needed).
- `fit_summary()` prints a training report (leaderboard, best model, feature importance availability, etc.).

> **AutoGluon internals:** It automatically does feature processing, trains multiple model types, and performs model selection/ensembling.

---

## 6) Prepare Test Data to Match Train Schema

```python
test_identity = pd.read_csv(directory+'test_identity.csv')
test_transaction = pd.read_csv(directory+'test_transaction.csv')
test_data = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
```

**Align columns to what the predictor actually trained on:**

```python
# --- ALIGN TEST COLUMNS TO TRAIN FEATURES ---
train_features = predictor.features()

# Add missing training features to test as NaN
missing_cols = [c for c in train_features if c not in test_data.columns]
for c in missing_cols:
    test_data[c] = np.nan

# Drop any extras not used during training (including label if present)
extra_cols = [c for c in test_data.columns if c not in train_features]
if extra_cols:
    test_data = test_data.drop(columns=extra_cols)

# Reorder columns to exactly match training
test_data = test_data[train_features]

print(f"Aligned test columns. Added {len(missing_cols)} missing, dropped {len(extra_cols)} extras.")
# -----------------------------------------------------------
```

- `predictor.features()` tells you the **exact** columns used during training after AutoGluon’s preprocessing.
- You:
  - Add any missing columns to test (filled with `NaN`).
  - Remove columns not used in training (ensures 1:1 schema).
  - Reorder columns to match training order.

> **Why this matters:** Mismatched columns will cause inference errors or wrong predictions.

---

## 7) Predict Probabilities

```python
y_predproba = predictor.predict_proba(test_data)
y_predproba.head(5)

predictor.positive_class

y_predproba = predictor.predict_proba(test_data, as_multiclass=False)
```

- `predict_proba` returns fraud **probabilities**.
- `predictor.positive_class` shows which class the positive label corresponds to (should be `1` for `isFraud`).
- `as_multiclass=False` ensures a 1D array of probabilities for the positive class (instead of a 2D array with both classes).

---

## 8) Create the Submission File

```python
submission = pd.read_csv(directory+'sample_submission.csv')
submission['isFraud'] = y_predproba
submission.head()
submission.to_csv(directory+'my_submission.csv', index=False)
```

- Loads the sample submission to ensure correct **row order** and **columns**.
- Replaces the `isFraud` column with your predicted probabilities.
- Writes `my_submission.csv` (ready to upload to the competition).

---

## Practical Tips & Gotchas

- **Sampling:** Great for quick experiments, but final models should train on **100%** of the data.
- **Imbalance:** Fraud datasets are typically imbalanced. AUC helps, but you may want to try:
  - `presets='best_quality'` (slower, often better).
  - Longer `time_limit`.
  - Custom hyperparameters or feature engineering.
- **Memory:** Large CSVs can be heavy. If you hit OOM:
  - Use chunked reading or cast dtypes (`category`, `float32`, `int32`).
  - Drop low-variance or ID-like columns.
- **Reproducibility:** Fix random seeds (`random_state`) where possible. Keep environment versions consistent.
- **Schema Drift:** New/old test sets must match the training feature set. Your alignment block handles this robustly.
- **Leaderboard vs Local AUC:** Expect some gap. Cross-validation helps estimate generalization.

---

## How to Extend This Pipeline

- **Feature Importance:** `predictor.feature_importance(train_data)` to understand key drivers.
- **Hyperparameter Tuning:** Use `hyperparameters` argument in `.fit(...)`.
- **Ensembling/Stacking:** AutoGluon already ensembles; experiment with presets and longer training time.
- **Inference Speed:** After training, you can `.persist_trainer()` or export the best model for deployment.

---

## Re-running on Full Data (Remove Sampling)

Replace the sampling block with:

```python
# Use full data (no downsampling)
# train_data = train_data  # do nothing
```

Or just delete that section entirely.

---

**End of walkthrough.** Happy modeling! 🚀
