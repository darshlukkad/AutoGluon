# California House Prices — AutoGluon Fast Baseline

A clean, reproducible baseline for the Kaggle competition **`california-house-prices`** using [AutoGluon Tabular](https://auto.gluon.ai/). The script is optimized for **Google Colab** and aims to get you from zero to a valid submission quickly, with sensible defaults, clear structure, and speed-focused settings.

> ✨ This repo prioritizes *speed* and *clarity*: single-model LightGBM, log1p preprocessing, deterministic splits, and a small time budget. It’s great for academic demos and fast iteration.

## Contents

- [What this script does](#what-this-script-does)
- [Quick start (Google Colab)](#quick-start-google-colab)
- [Configuration knobs](#configuration-knobs)
- [Data & preprocessing](#data--preprocessing)
- [Modeling & training](#modeling--training)
- [Evaluation & metrics](#evaluation--metrics)
- [Submission](#submission)
- [Reproducibility](#reproducibility)
- [Speed vs. quality: tuning tips](#speed-vs-quality-tuning-tips)
- [Local setup (optional)](#local-setup-optional)
- [Troubleshooting](#troubleshooting)
- [Notes & acknowledgements](#notes--acknowledgements)
- [License](#license)

## What this script does

1. **Installs** dependencies: `kaggle`, `autogluon.tabular`, `scikit-learn`  
2. **Authenticates** the Kaggle CLI using your `kaggle.json` API token.  
3. **Downloads & unzips** the competition files to a chosen data directory.  
4. **Loads** `train.csv`, `test.csv`, and `sample_submission.csv`.  
5. **Preprocesses** features:
   - drops obvious ID columns (`Id`/`id`) if present,
   - `log1p` transforms **all numeric features** (non-negative clipped),
   - `log1p` transforms the **target** (`"Sold Price"`) for stability,
   - downcasts numerics to `float32` for speed/memory.
6. **Splits** the data (deterministic):  
   - 10% **final holdout** (never seen during training),  
   - from the remaining 90%, 10% becomes a **dev/validation** set.  
   - Effective split ≈ **81% train / 9% dev / 10% holdout**.
7. **Trains** a **single LightGBM model** via AutoGluon with a tight **time budget (`TL`)**.
8. **Evaluates** on the holdout set (log-RMSE) and also prints **dollar-scale RMSE**.
9. **Predicts** on `test.csv` and writes a **`submission.csv`** in the model folder.
10. **Submits** to Kaggle via CLI and lists your latest submissions.

## Quick start (Google Colab)

1. Open a new Colab notebook and paste the script from this repo.
2. Make sure you have your `kaggle.json` API token handy.
   - In Colab: **Files → Upload** `kaggle.json` to `/content/` (matches the script), **or** adjust the path.
3. Run the notebook cells in order. You should see output like:
   - competition files downloaded,
   - model training logs,
   - holdout metrics,
   - path to `submission.csv`,
   - Kaggle submission result.

> **Important**: The script currently copies `kaggle.json` from `/content/kaggle.json`. If you keep your token in Google Drive, change the copy command to point to your Drive path (e.g., `/content/drive/MyDrive/kaggle/kaggle.json`).

## Configuration knobs

These are defined near the top of the script:

```python
KAGGLE_COMPETITION = "california-house-prices"
DATA_DIR = "/content/data"
DATASET = os.path.join(DATA_DIR, KAGGLE_COMPETITION)
AUTOGLUON_SAVE_PATH = os.path.join(DATA_DIR, "AutoGluonModels")
