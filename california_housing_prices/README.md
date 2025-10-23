# California House Prices — AutoGluon (Colab) ⚡

A fast, reproducible pipeline to:
1) download a Kaggle competition dataset,  
2) train a lightweight AutoGluon (LightGBM-only) regressor under a tight time budget,  
3) evaluate on a holdout split, and  
4) generate & submit `submission.csv` to Kaggle — all from Google Colab.

---

## What this script does

- **Setup & config**
  - Defines competition name (`california-house-prices`), data dirs, and AutoGluon model path.
  - Installs `kaggle`, `autogluon.tabular`, and `scikit-learn`.
  - Mounts Google Drive (assumes you’ll provide `kaggle.json` creds).

- **Data acquisition**
  - Copies `kaggle.json` into `~/.kaggle/` and sets permissions.
  - Downloads the competition ZIP to `DATA_DIR`, unzips to `DATASET`, and lists files.

- **Fast training loop (demo-friendly)**
  - Loads `train.csv`, `test.csv`, and `sample_submission.csv`.
  - **Speed tricks**:
    - Caps training rows to `N_TRAIN = 20_000` (deterministic sample).
    - Uses **LightGBM only** for rapid training.
    - Sets `time_limit = TL` (default 300s) and uses all available CPU threads.
  - **Preprocessing**:
    - Drops obvious ID columns (`Id`/`id`) if present.
    - Applies **log1p** to numeric features and **target** (`Sold Price`), then downcasts to `float32`.
  - **Splits**:
    - 10% **final holdout** (never seen during training).
    - From the remaining 90%, uses 10% as **dev/tuning** data.
  - **Training**:
    - AutoGluon `TabularPredictor` with `presets="good_quality"`, LightGBM hyperparams, fixed seeds.
    - No bagging / stacking to keep things fast.
  - **Evaluation & export**:
    - Prints **log-scale RMSE** on the holdout.
    - Also computes **dollar-scale RMSE** by inverting the log1p transform.
    - Predicts on test, writes **`submission.csv`** under `AutoGluonModels/<run>/submission.csv`.

- **Submission to Kaggle**
  - Picks the most recent `submission.csv`.
  - Runs `kaggle competitions submit` with a short message.
  - Shows the latest submissions table (top lines).

---

## Requirements

- **Google Colab** runtime (recommended).
- A valid **Kaggle API token** (`kaggle.json`) with competition access.
- Enough disk space in Colab (the dataset is auto-downloaded).

---

## Quickstart (Colab)

1. **Open a Colab notebook** and paste the script.
2. **Add your Kaggle API token**:
   - Download `kaggle.json` from your Kaggle account (Account → API → Create New Token).
   - Upload it to Colab root as `/content/kaggle.json` (the script expects this exact path).
3. **Run all cells**:
   - The script will install packages, mount Drive, download data, train, evaluate, save `submission.csv`, and submit to Kaggle.

> If Drive mounting prompts appear, accept them. If the competition is private or requires rules acceptance, do that on Kaggle first.

---

## Key parameters you might tweak

| Variable | What it controls | Default |
|---|---|---|
| `KAGGLE_COMPETITION` | Kaggle competition slug | `"california-house-prices"` |
| `DATA_DIR` | Root data folder | `"/content/data"` |
| `AUTOGLUON_SAVE_PATH` | Where models & outputs go | `"/content/data/AutoGluonModels"` |
| `SEED` | Reproducibility | `42` |
| `N_TRAIN` | Row cap for speed | `20_000` |
| `TL` | Training time limit (seconds) | `300` |
| `TARGET` | Target column | `"Sold Price"` |

---

## Model & preprocessing choices (brief)

- **LightGBM-only**: `GBM` with modest rounds and early stopping → fast, strong baseline.
- **Log1p transforms**: Stabilizes skewed numeric features and the target; predictions are inverted via `expm1`.
- **No bagging/stacking**: Minimizes training time and memory for a quick academic demo.
- **Threading**: Uses available CPU cores for faster training.

---

## Outputs

- **Models + artifacts**: `AutoGluonModels/<run>/`
- **Submission file**: `AutoGluonModels/<run>/submission.csv`
- **Printed metrics**:
  - Log-scale RMSE from AutoGluon.
  - Inverted **$-scale RMSE** (easier to interpret).
- **Kaggle submission**: Performed via CLI; a short summary table is shown.

---

## Common issues & tips

- **\`403\`/\`404\` on download**: Ensure you joined the competition and accepted rules.
- **\`kaggle.json\` not found**: Confirm it exists at `/content/kaggle.json` before running.
- **Time limit too tight**: Increase `TL` (e.g., `600–1200`) for better accuracy.
- **Memory/timeouts**: Lower `N_TRAIN` or keep LightGBM-only (already done here).

---

## Customize further

- Add more models: enable `"CAT"`, `"XGB"`, `"RF"`, etc. in `hyperparameters`.
- Improve quality: try `presets="medium_quality"` (increase `TL` accordingly).
- Feature engineering: add domain features before training.
- Validation scheme: replace the simple split with K-Fold for more robust estimates.

---

**License**: Use and modify freely for educational or competition purposes.
