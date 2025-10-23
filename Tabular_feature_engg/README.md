# AutoGluon Feature Generators & Mixed‑Type Data — Code Walkthrough

This document explains what each part of your script does and why. It focuses on **feature generation** in AutoGluon when handling mixed data types (numeric, integers, datetimes, categoricals, and free‑text).

---

## TL;DR

- You **synthesize a regression dataset** and enrich it with additional columns of various data types.
- You use **`AutoMLPipelineFeatureGenerator`** to automatically process mixed types into model‑ready features.
- You train a **`TabularPredictor`** (LightGBM via `'GBM'`) with a specific feature generator.
- You then **change dtypes**, **introduce missing values**, and **re‑fit** feature generators to see how they behave.
- Finally, you build a **custom `PipelineFeatureGenerator`** that passes through numeric features and encodes categoricals with a small cap (`maximum_num_cat=10`).

---

## 1) Install & Imports

```python
%pip install autogluon

from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import numpy as np
import random
from sklearn.datasets import make_regression
from datetime import datetime
```

- Installs AutoGluon and brings in:
  - `TabularDataset` / `TabularPredictor` for high‑level tabular ML.
  - `pandas`/`numpy` for data wrangling.
  - `make_regression` to generate a synthetic regression dataset.

---

## 2) Create a Synthetic Dataset

```python
x, y = make_regression(n_samples=100, n_features=5, n_targets=1, random_state=1)
dfx = pd.DataFrame(x, columns=['A','B','C','D','E'])
dfy = pd.DataFrame(y, columns=['label'])
```

- `x` contains 5 numerical features; `y` is a continuous target (`label`).
- `random_state=1` makes the synthetic data reproducible.

---

## 3) Enrich Columns to Demonstrate Dtype Handling

```python
# Integer column
dfx['B'] = dfx['B'].astype(int)

# Datetime column (fixed base + days)
dfx['C'] = datetime(2000,1,1) + pd.to_timedelta(dfx['C'].astype(int), unit='D')

# Categorical column via binning
dfx['D'] = pd.cut(dfx['D'] * 10, [-np.inf, -5, 0, 5, np.inf], labels=['v','w','x','y'])

# Short free-text column
dfx['E'] = pd.Series(
    list(' '.join(random.choice(["abc", "d", "ef", "ghi", "jkl"]) for _ in range(4))
         for _ in range(100))
)
```

- **B** becomes an **integer**.
- **C** becomes a **datetime** (derived from a baseline date).
- **D** becomes a **categorical** with 4 bins (`v/w/x/y`).
- **E** becomes a **short text** field made from random tokens.

> Repro tip: set `random.seed(0)` so the text column is reproducible.

---

## 4) Wrap as a `TabularDataset`

```python
dataset = TabularDataset(dfx)
print(dfx)
```

- `TabularDataset` is a thin wrapper around a pandas DataFrame; it helps AutoGluon infer types and perform validations.

---

## 5) Default AutoML Feature Pipeline

```python
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
auto_ml_pipeline_feature_generator.fit_transform(X=dfx)
```

- **What it does:** Automatically **detects dtypes** and applies appropriate transforms:
  - **Numeric**: pass‑through or lightweight normalization.
  - **Categorical**: encodings (like one‑hot / ordinal depending on downstream model).
  - **Datetime**: extract year, month, day, dayofweek, etc.
  - **Text**: converts to bag‑of‑words/character‑ngrams, token counts, and other text stats.
  - **Missing values**: adds indicator features and safe imputations.
- `fit_transform` both learns metadata from `X` and returns the transformed feature matrix.

---

## 6) Train a Predictor with a Specific Feature Generator

```python
df = pd.concat([dfx, dfy], axis=1)
predictor = TabularPredictor(label='label')
predictor.fit(df, hyperparameters={'GBM': {}}, feature_generator=auto_ml_pipeline_feature_generator)
```

- Combines features (`dfx`) with the target (`dfy`) into `df`.
- Creates a **`TabularPredictor`** to model `label`.
- **`hyperparameters={'GBM': {}}`** constrains training to Gradient‑Boosted Trees (LightGBM family) instead of the full AutoGluon model zoo.
- **`feature_generator=...`** tells the predictor which pipeline to use. AutoGluon will **clone and (re)fit** it internally on the training data, ensuring it becomes part of the saved model pipeline for consistent inference.

> ⚠️ Good practice: Pass an **unfitted** generator (as you did initially) and let the `predictor` handle fitting. Avoid reusing a generator that has already been fit on data it shouldn’t see (to prevent subtle leakage in other contexts).

---

## 7) Change a Dtype and Re‑fit a New Generator

```python
dfx["B"] = dfx["B"].astype("category")
auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
auto_ml_pipeline_feature_generator.fit_transform(X=dfx)
```

- Now **B** is **categorical** instead of integer. The feature generator will treat it accordingly (e.g., one‑hot/ordinal), which can change downstream model behavior.

> Schema changes between train and inference can cause drift. In production, keep dtypes **consistent** between training and serving.

---

## 8) Introduce Missing Values and Re‑fit Again

```python
dfx.iloc[0] = np.nan
dfx.head()

auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
auto_ml_pipeline_feature_generator.fit_transform(X=dfx)
```

- The **first row** is now entirely `NaN`. AutoGluon’s pipeline is built to **handle missingness** by imputing and/or adding **is‑missing indicators** so models can learn from missing patterns.

---

## 9) Build a Custom Feature Pipeline

```python
from autogluon.features.generators import PipelineFeatureGenerator, CategoryFeatureGenerator, IdentityFeatureGenerator
from autogluon.common.features.types import R_INT, R_FLOAT

mypipeline = PipelineFeatureGenerator(
    generators=[[
        CategoryFeatureGenerator(maximum_num_cat=10),  # limit unique categories processed
        IdentityFeatureGenerator(infer_features_in_args=dict(valid_raw_types=[R_INT, R_FLOAT])),
    ]]
)

mypipeline.fit_transform(X=dfx)
```

- **`PipelineFeatureGenerator`** orchestrates sub‑generators.
- The inner list `[...]` indicates **generators applied in parallel to the same inputs** at that stage:
  - `CategoryFeatureGenerator(maximum_num_cat=10)`: encodes categoricals, but **caps** the number of categories to avoid huge feature blow‑ups (rare categories may be grouped).
  - `IdentityFeatureGenerator(...)`: simply **passes through** features whose raw types match **`R_INT`** or **`R_FLOAT`** (i.e., genuine numeric columns) with minimal changes.
- `fit_transform` returns the final, concatenated feature matrix produced by the pipeline.

> You can chain multiple stages (outer list with multiple elements) if you want to, for example, generate features, then derive interactions, etc.

---

## Practical Tips & Gotchas

- **Consistency is king:** Keep data types stable between training and inference. If you train with `B` as integer, don’t serve with `B` as categorical.
- **Text features:** Even short text like in **E** can add predictive signal; AutoGluon handles tokenization and n‑grams automatically for many models.
- **Missing values:** Deliberately test pipelines with missingness (as you did) to verify robustness.
- **Feature importance:** Use `predictor.feature_importance(df)` to understand which engineered features matter.
- **Reproducibility:** Set both `numpy` and Python `random` seeds when generating synthetic data and text.
- **Scope of generators:** Prefer providing an unfitted generator to `predictor.fit(...)`, and let AutoGluon manage it and persist it with the model.

---

## Where to Go Next

- Try different hyperparameters: `hyperparameters={'GBM': {'num_boost_round': 200}}`.
- Use more models (e.g., `'CAT'`, `'RF'`, `'NN_TORCH'`) by expanding the `hyperparameters` dict.
- Add time budgets: `predictor.fit(..., time_limit=600)`.
- Explore `predictor.leaderboard()` and `predictor.persist_trainer()` for diagnostics and deployment.

---

**End of walkthrough.** Have fun experimenting with feature generators! 🔧📊
