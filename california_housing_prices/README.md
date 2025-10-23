# California House Prices — AutoGluon Fast Baseline

A clean, reproducible baseline for the Kaggle competition **`california-house-prices`** using [AutoGluon Tabular](https://auto.gluon.ai/). The script is optimized for **Google Colab** and aims to get you from zero to a valid submission quickly, with sensible defaults, clear structure, and speed-focused settings.

> ✨ This repo prioritizes *speed* and *clarity*: single-model LightGBM, log1p preprocessing, deterministic splits, and a small time budget. It’s great for academic demos and fast iteration.

## Contents

- [What this script does](#what-this-script-does)
- [Google Colab: End-to-End Guide](#google-colab-end-to-end-guide)
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

1. **Installs** dependencies: `kaggle`, `autogluon.tabular`, `scikit-learn`.  
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

---

## Google Colab: End-to-End Guide

This project is designed to run smoothly on **Google Colab**. Here’s everything you need to know to go from a blank notebook to a Kaggle submission:

### 1) Prepare your Kaggle API token

- Go to **Kaggle → Account → API → Create New Token**.  
- You’ll download a file named **`kaggle.json`**. Keep it handy.

### 2) Choose where to put `kaggle.json` in Colab

You have two easy options—**either is fine**. The script supports both:

**Option A — Upload directly to `/content` (fastest):**
1. In Colab, click the **folder** icon (left sidebar) → **Upload** → pick `kaggle.json`.
2. It will appear as `/content/kaggle.json`.

**Option B — Store in Google Drive (persists across sessions):**
1. Put `kaggle.json` somewhere in Drive, e.g. `MyDrive/kaggle/kaggle.json`.
2. The script mounts Drive and you can copy from there (see next steps).

> The default script copies from `/content/kaggle.json`. If you prefer Drive, change the copy command to the Drive path (example below).

### 3) Install packages (first notebook cell)

```python
!pip install -q kaggle autogluon.tabular scikit-learn
