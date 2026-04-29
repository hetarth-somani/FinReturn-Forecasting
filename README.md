# FinReturn Forecasting

A research-backed machine learning pipeline for predicting financial asset returns in low signal-to-noise ratio environments. Built with PyTorch and scikit-learn.

## Highlights

- **Supervised Autoencoder MLP** — Jointly learns compressed feature representations and predictions through a shared encoder, inspired by the [Jane Street Market Prediction](https://www.kaggle.com/c/jane-street-market-prediction) 1st place solution.
- **Temporal Cross-Validation** — Uses `TimeSeriesSplit` with a configurable gap to strictly prevent future data leakage, a critical requirement for financial time-series.
- **Recency Weighting** — Applies exponential decay sample weights so the model prioritizes learning from recent market regimes, based on findings from [Timmermann & Granger (2004)](https://doi.org/10.1016/S0927-5398(04)00032-5).
- **SWA + Mixup + Huber Loss** — A triple-regularization stack that finds flat, generalizable loss minima in extremely noisy data.
- **Equal-Weight Ensembling** — Blends a neural network (AE-MLP) with a Ridge Regression baseline using simple averaging, avoiding the data-mining trap of optimized blend weights.
- **Adversarial Validation** — A built-in diagnostic that detects train/test distribution shift before training begins.

## Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy pandas scikit-learn lightgbm scipy
```

### 2. Prepare Your Data

Place your training and test data (Parquet or CSV) in the `data/` directory. The pipeline expects:
- A **training file** with feature columns and a target column.
- A **test file** with the same feature columns (target column absent).

### 3. Configure

Open `src/train.py` and edit the `CONFIG` dictionary at the top:

```python
CONFIG = {
    "train_path": "data/train.parquet",   # Your training data
    "test_path":  "data/test.parquet",    # Your test data
    "output_path": "output/submission.csv",
    "target_col": "TARGET",               # Name of your target column
    "id_col": "ID",                       # ID column (or None)
    "exclude_cols": ["ID"],               # Columns to drop from features
    "seeds": [42, 123, 456],              # Random seeds for ensembling
    ...
}
```

### 4. Run

```bash
python src/train.py
```

The pipeline will:
1. Run adversarial validation to check for distribution shift.
2. Train a Supervised Autoencoder MLP across multiple seeds and temporal folds.
3. Train a Ridge Regression baseline for diversity.
4. Blend predictions and apply shrinkage tuning.
5. Save the final predictions to `output/submission.csv`.

## Project Structure

```
├── src/
│   ├── train.py                    # Main training pipeline
│   ├── data_loader.py              # Data loading and memory optimization
│   ├── eda.py                      # Exploratory data analysis & plotting
│   ├── feature_engineering.py      # Price trajectory & lag feature engineering
│   ├── regime_detection.py         # Unsupervised market regime clustering
│   └── adversarial_validation.py   # Train/test distribution shift detection
├── data/                           # Your dataset files (not tracked)
├── output/                         # Generated predictions
└── README.md
```

## Key References

| Source | Contribution |
|---|---|
| [Timmermann & Granger (2004)](https://doi.org/10.1016/S0927-5398(04)00032-5) | Temporal validation, recency weighting, equal-weight ensembling |
| [Jane Street 1st Place](https://www.kaggle.com/c/jane-street-market-prediction/discussion/224348) | Supervised Autoencoder MLP architecture |
| [Optiver 2nd Place](https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/274970) | 1D-CNN + diverse ensembling strategy |

## License

MIT
