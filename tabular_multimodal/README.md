# AutoGluon Tabular — PetFinder Multimodal Tutorial: Code Walkthrough

This guide explains each step of your script, focusing on how **AutoGluon Tabular** handles **multimodal** data (tabular + text + images).

---

## TL;DR

1. Install AutoGluon and **download** a prepared PetFinder dataset (CSV + images).
2. **Load** train/dev CSVs and **clean** the image-path column.
3. **Preview** a sample description and image for sanity checks.
4. **Sample** 500 rows to speed up the demo.
5. Build **FeatureMetadata** that marks the image column as `image_path`.
6. Use `get_hyperparameter_config('multimodal')` to enable **multimodal training**.
7. Train a `TabularPredictor` with `time_limit=900` seconds and view the **leaderboard**.

---

## 1) Install & Download Data

```python
!pip install autogluon

download_dir = './ag_petfinder_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/petfinder_kaggle.zip'

from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)
import os
os.listdir(download_dir)
```

- Installs **AutoGluon** (Tabular + Multimodal engines).
- Downloads and unzips a prepared PetFinder dataset into `./ag_petfinder_tutorial`.
- `os.listdir(...)` confirms the download contents.

---

## 2) Locate the Dataset Folders

```python
dataset_path = download_dir + '/petfinder_processed'
os.listdir(dataset_path)
os.listdir(dataset_path + '/train_images')[:10]
```

- `petfinder_processed/` contains:
  - `train.csv`, `dev.csv` (aka validation/test split for the tutorial).
  - `train_images/` with pet photos referenced by the CSVs.

---

## 3) Load CSVs

```python
import pandas as pd

train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data  = pd.read_csv(f'{dataset_path}/dev.csv',   index_col=0)

train_data.head(3)
```

- Loads **training** and **dev** (used here as test) splits.
- `index_col=0` uses the first column as index (often an ID).

---

## 4) Identify Label & Image Column

```python
label = 'AdoptionSpeed'
image_col = 'Images'
```

- **Target** is `AdoptionSpeed` (ordinal classes 0–4 in PetFinder).
- **Image file paths** are in the `Images` column. Often multiple paths are **semicolon-separated**.

---

## 5) Keep Only the First Image per Row

```python
train_data[image_col].iloc[0]
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
test_data[image_col]  = test_data[image_col].apply(lambda ele: ele.split(';')[0])
train_data[image_col].iloc[0]
```

- To simplify, we retain **only the first image** (many rows list multiple images).
- This keeps training lightweight while still leveraging image signal.

---

## 6) Convert to Absolute Paths

```python
import os

def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col]  = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))

train_data[image_col].iloc[0]
```

- Many training pipelines expect **absolute paths**. This helper:
  1) Joins each relative path with `dataset_path`.
  2) Converts to an absolute path via `os.path.abspath`.
- Ensures AutoGluon can **actually find the image files** during training.

---

## 7) Quick Sanity Checks (Text + Image)

```python
train_data.head(3)

example_row = train_data.iloc[1]
example_row['Description']

from IPython.display import Image, display
example_image = example_row['Images']
pil_img = Image(filename=example_image)
display(pil_img)
```

- Inspect the **`Description`** text and **display** the corresponding image to confirm the paths are valid.

---

## 8) Subsample for a Faster Demo

```python
train_data = train_data.sample(500, random_state=0)
```

- Keeps 500 rows to speed up the tutorial (good for CPU-only sessions).
- For best accuracy, **remove** this sampling and train on the full dataset.

---

## 9) Feature Metadata (Tell AutoGluon About Images)

```python
from autogluon.tabular import FeatureMetadata

feature_metadata = FeatureMetadata.from_df(train_data)
print(feature_metadata)

feature_metadata = feature_metadata.add_special_types({image_col: ['image_path']})
print(feature_metadata)
```

- **FeatureMetadata** helps AutoGluon understand special columns.
- Marking the `Images` column as **`image_path`** signals that this column points to image files that should be consumed by the **multimodal** model (AutoMM engine under the hood).

---

## 10) Multimodal Hyperparameters

```python
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
hyperparameters = get_hyperparameter_config('multimodal')
hyperparameters
```

- Retrieves a preset **multimodal** config that enables models capable of using **text, images, and tabular** features together.
- Internally, AutoGluon leverages its **AutoMM** stack for image/text encoders and fuses them with tabular models.

---

## 11) Train the Predictor

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label=label).fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
    feature_metadata=feature_metadata,
    time_limit=900,
)
```

- Trains a single unified **TabularPredictor** that:
  - Reads image **paths** (`image_path`), **text** (`Description`), and standard tabular features.
  - Uses the **multimodal** preset to pick/fit suitable encoders.
  - Runs within a **15-minute** time limit.

> **Tips**
> - If you have a **GPU**, training will be significantly faster and stronger for image/text encoders.
> - Increase `time_limit` and drop the 500-row sampling for better performance.
> - Ensure `torch`/`torchvision` are installed if the environment doesn’t include them by default.

---

## 12) Evaluate Models with a Leaderboard

```python
leaderboard = predictor.leaderboard(test_data)
```

- Shows a ranked table of models, fit/predict times, and scores.
- Since `test_data` includes ground-truth labels, this leaderboard will reflect **hold-out performance** on the dev split.

---

## Troubleshooting & Best Practices

- **Missing files**: If image display fails, print a few `train_data[image_col].head()` and verify paths exist on disk.
- **Paths on Windows**: Use `os.path.join` everywhere to avoid separator issues.
- **Multiple images**: You can keep multiple paths per row if desired; ensure the model/hyperparameters support multi-image inputs.
- **Metrics**: `AdoptionSpeed` is ordinal (0–4). Consider metrics aligned with ordinal targets or frame it as multiclass classification.
- **Reproducibility**: Control randomness with `random_state` and use fixed package versions.

---

**End of walkthrough.** You now have a working pipeline that fuses **tabular**, **text**, and **image** signals with AutoGluon. 🐶📷📊
