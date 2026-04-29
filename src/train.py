"""
Financial Return Forecasting — Training Pipeline

A robust, research-backed pipeline for predicting financial asset returns 
in noisy, low signal-to-noise ratio environments.

Architecture:
  - Supervised Autoencoder MLP (inspired by Jane Street 1st place solution)
  - Temporal Cross-Validation via TimeSeriesSplit (prevents future data leakage)
  - Exponential decay recency weighting (Timmermann & Granger, 2004)
  - SWA (Stochastic Weight Averaging) + Mixup + Huber Loss
  - Ridge Regression baseline for ensemble diversity
  - Equal-weight ensemble + conservative prediction shrinkage

References:
  - Timmermann & Granger (2004), "In-sample vs. out-of-sample tests of
    stock return predictability in the context of data mining"
  - Jane Street Market Prediction, 1st Place Solution (Supervised Autoencoder)
  - Optiver Realized Volatility, 2nd Place Solution (1D-CNN + Ensembling)
"""

import os
import gc
import time
import datetime
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

# ====================================================================
# USER CONFIGURATION — Edit this section for your own dataset
# ====================================================================
CONFIG = {
    # --- Data paths ---
    "train_path": "data/train.parquet",       # Path to training data (Parquet or CSV)
    "test_path":  "data/test.parquet",         # Path to test data (Parquet or CSV)
    "output_path": "output/submission.csv",    # Where to save predictions

    # --- Column names ---
    "target_col": "TARGET",                    # Name of the target column in train
    "id_col": "ID",                            # Name of the ID column (set to None if absent)
    "exclude_cols": ["ID", "Id", "index",      # Columns to exclude from features
                     "timestamp", "time_id"],   

    # --- Pseudo-labeling (optional) ---
    "pseudo_label_path": None,                 # Path to a previous submission CSV (or None)
    "pseudo_weight": 0.3,                      # Weight for pseudo-labeled samples

    # --- Training hyperparameters ---
    "seeds": [42, 123, 456],                   # Random seeds for ensemble diversity
    "n_folds": 5,                              # Number of temporal CV folds
    "batch_size": 1024,                        # Batch size (1024 acts as regularizer)
    "max_epochs": 120,                         # Maximum training epochs per fold
    "patience": 20,                            # Early stopping patience
    "swa_start": 40,                           # Epoch to begin SWA averaging
    "ts_gap": 50,                              # Gap rows between train/val in TimeSeriesSplit
    "learning_rate": 1e-3,
    "weight_decay": 1e-3,
}


def load_data(path):
    if path.endswith('.csv'):
        return pd.read_csv(path)
    return pd.read_parquet(path)


# ====================================================================
# MODEL DEFINITIONS
# ====================================================================
class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std
    def forward(self, x):
        if self.training and self.std > 0:
            return x + torch.randn_like(x) * self.std
        return x

class SupervisedAutoencoderMLP(nn.Module):
    """
    Jointly trains an autoencoder for feature extraction and an MLP for
    prediction. The encoder's learned representation is concatenated with
    the original features before being fed into the prediction head.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.noise = GaussianNoise(std=0.1)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.2)
        )
        self.decoder = nn.Linear(128, input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 128, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x_noisy = self.noise(x)
        encoded = self.encoder(x_noisy)
        decoded = self.decoder(encoded)
        concat = torch.cat([x, encoded], dim=1)
        pred = self.mlp(concat).squeeze(-1)
        return pred, decoded

def mixup_data(x, y, w, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx], lam * w + (1 - lam) * w[idx]


# ====================================================================
# TRAINING ENGINE
# ====================================================================
def train_ae_mlp(t_X_scaled, t_y, t_recency, t_X_pseudo, t_y_pseudo,
                 tr_idx, va_idx, seed, cfg, device, input_dim):
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = SupervisedAutoencoderMLP(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    criterion_pred = nn.HuberLoss(reduction='none')
    criterion_recon = nn.MSELoss()
    amp_scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    t_X_tr, t_y_tr = t_X_scaled[tr_idx], t_y[tr_idx]
    t_w_tr = t_recency[tr_idx]
    t_X_va, t_y_va = t_X_scaled[va_idx], t_y[va_idx]

    if t_X_pseudo is not None and len(t_X_pseudo) > 0:
        t_X_comb = torch.cat([t_X_tr, t_X_pseudo], dim=0)
        t_y_comb = torch.cat([t_y_tr, t_y_pseudo], dim=0)
        pseudo_w = torch.full((len(t_X_pseudo),), cfg["pseudo_weight"], device=device)
        t_w_comb = torch.cat([t_w_tr, pseudo_w], dim=0)
    else:
        t_X_comb, t_y_comb, t_w_comb = t_X_tr, t_y_tr, t_w_tr

    num_samples = len(t_X_comb)
    batch_size = cfg["batch_size"]

    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=5e-4, anneal_epochs=5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["swa_start"])

    best_val_rmse = float('inf')
    best_state = None
    patience_cnt = 0
    alpha_recon = 0.1

    for epoch in range(cfg["max_epochs"]):
        model.train()
        perm = torch.randperm(num_samples, device=device)

        for i in range(0, num_samples, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb, wb = t_X_comb[idx], t_y_comb[idx], t_w_comb[idx]
            xb_mix, yb_mix, wb_mix = mixup_data(xb, yb, wb, alpha=0.2)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                pred, recon = model(xb_mix)
                loss_pred = (criterion_pred(pred, yb_mix) * wb_mix).mean()
                loss_recon = criterion_recon(recon, xb_mix)
                loss = loss_pred + alpha_recon * loss_recon

            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()

        if epoch >= cfg["swa_start"]:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        eval_model = swa_model if epoch >= cfg["swa_start"] else model
        eval_model.eval()
        with torch.no_grad():
            val_preds = []
            for i in range(0, len(t_X_va), batch_size):
                vp, _ = eval_model(t_X_va[i:i + batch_size])
                val_preds.append(vp)
            val_pred = torch.cat(val_preds)
            val_rmse = torch.sqrt(torch.mean((val_pred - t_y_va) ** 2)).item()

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            src = swa_model.module if epoch >= cfg["swa_start"] else model
            best_state = {k: v.cpu().clone() for k, v in src.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                break

    final_model = SupervisedAutoencoderMLP(input_dim).to(device)
    final_model.load_state_dict(best_state)
    bn_dl = DataLoader(TensorDataset(t_X_tr.cpu(), t_y_tr.cpu()), batch_size=batch_size, shuffle=True)
    update_bn(bn_dl, final_model, device=device)
    final_model.eval()
    return final_model


def predict_gpu(model, t_X, batch_size):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(t_X), batch_size):
            p, _ = model(t_X[i:i + batch_size])
            preds.append(p)
    return torch.cat(preds).cpu().numpy()


# ====================================================================
# MAIN PIPELINE
# ====================================================================
def main(cfg=None):
    if cfg is None:
        cfg = CONFIG

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- 1. Load Data ---
    print("\n[1/8] Loading data...")
    df_train = load_data(cfg["train_path"])
    df_test  = load_data(cfg["test_path"])

    exclude = [cfg["target_col"]] + (cfg["exclude_cols"] or [])
    features = [c for c in df_train.columns if c not in exclude]
    
    X = df_train[features].values.astype(np.float32)
    y = df_train[cfg["target_col"]].values.astype(np.float32)
    X_test = df_test[features].values.astype(np.float32)
    
    id_col = cfg.get("id_col")
    test_ids = df_test[id_col].values if id_col and id_col in df_test.columns else np.arange(len(X_test))
    input_dim = X.shape[1]

    print(f"Train: {X.shape}, Test: {X_test.shape}, Features: {len(features)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    # Recency weights (Timmermann & Granger)
    recency_weights = np.exp(np.linspace(-1, 0, len(y))).astype(np.float32)

    # Pseudo labels (optional)
    t_X_pseudo, t_y_pseudo = None, None
    if cfg.get("pseudo_label_path") and os.path.exists(cfg["pseudo_label_path"]):
        pseudo_df = pd.read_csv(cfg["pseudo_label_path"])
        pseudo_targets = pseudo_df[cfg["target_col"]].values.astype(np.float32)
        pred_std = np.abs(pseudo_targets - pseudo_targets.mean())
        mask = pred_std <= np.percentile(pred_std, 75)
        X_pseudo = X_test_scaled[mask]
        y_pseudo = pseudo_targets[mask]
        print(f"Pseudo labels: {mask.sum()}/{len(mask)} kept")
        t_X_pseudo = torch.FloatTensor(X_pseudo).to(device)
        t_y_pseudo = torch.FloatTensor(y_pseudo).to(device)

    # --- 2. Adversarial Validation ---
    print("\n[2/8] Running Adversarial Validation...")
    np.random.seed(42)
    sample = 50000
    idx_tr = np.random.choice(len(X_scaled), min(len(X_scaled), sample), replace=False)
    idx_te = np.random.choice(len(X_test_scaled), min(len(X_test_scaled), sample), replace=False)
    X_adv = np.vstack([X_scaled[idx_tr], X_test_scaled[idx_te]])
    y_adv = np.hstack([np.zeros(len(idx_tr)), np.ones(len(idx_te))])
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    auc = cross_val_score(clf, X_adv, y_adv, cv=3, scoring='roc_auc').mean()
    print(f"Adversarial AUC: {auc:.4f}" + (" ⚠️ Shift detected!" if auc > 0.70 else " ✅ Aligned."))

    # --- 3. Pre-load tensors ---
    print("\n[3/8] Pre-loading tensors...")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    t_X_scaled = torch.FloatTensor(X_scaled).to(device)
    t_y = torch.FloatTensor(y).to(device)
    t_recency = torch.FloatTensor(recency_weights).to(device)
    t_X_test = torch.FloatTensor(X_test_scaled).to(device)

    # --- 4. AE-MLP Training ---
    seeds = cfg["seeds"]
    n_folds = cfg["n_folds"]
    batch_size = cfg["batch_size"]
    cv = TimeSeriesSplit(n_splits=n_folds, gap=cfg["ts_gap"])

    print(f"\n[4/8] Training AE-MLP ({len(seeds)} seeds × {n_folds} folds, TimeSeriesSplit)...")
    oof_mlp = np.full(len(X), np.nan, dtype=np.float32)
    test_mlp = np.zeros(len(X_test), dtype=np.float32)
    fold_counts = np.zeros(len(X), dtype=np.int32)

    start_time = time.time()
    for seed in seeds:
        print(f"\n▶ SEED {seed}")
        test_seed = np.zeros(len(X_test), dtype=np.float32)

        for fold, (tr_idx, va_idx) in enumerate(cv.split(X_scaled)):
            fold_start = time.time()
            model = train_ae_mlp(t_X_scaled, t_y, t_recency, t_X_pseudo, t_y_pseudo,
                                 tr_idx, va_idx, seed=seed + fold, cfg=cfg, device=device, input_dim=input_dim)

            o_p = predict_gpu(model, t_X_scaled[va_idx], batch_size)
            t_p = predict_gpu(model, t_X_test, batch_size)

            for i, vi in enumerate(va_idx):
                if np.isnan(oof_mlp[vi]):
                    oof_mlp[vi] = 0.0
                oof_mlp[vi] += o_p[i]
                fold_counts[vi] += 1

            test_seed += t_p / n_folds
            elapsed = time.time() - fold_start
            print(f"  Fold {fold + 1} (train={len(tr_idx):,}, val={len(va_idx):,}): "
                  f"R²={r2_score(y[va_idx], o_p):.5f} ({elapsed:.1f}s)")

            del model; gc.collect()
            if device.type == 'cuda': torch.cuda.empty_cache()

        test_mlp += test_seed / len(seeds)

        # Checkpoint after each seed
        os.makedirs(os.path.dirname(cfg["output_path"]) or '.', exist_ok=True)
        np.save(cfg["output_path"].replace('.csv', '_oof_ckpt.npy'), oof_mlp)
        np.save(cfg["output_path"].replace('.csv', '_test_ckpt.npy'), test_mlp)
        print(f"  [Checkpoint saved]")

    valid_mask = fold_counts > 0
    oof_mlp[valid_mask] /= fold_counts[valid_mask]
    print(f"\nAE-MLP OOF R²: {r2_score(y[valid_mask], oof_mlp[valid_mask]):.5f}")
    print(f"Training time: {datetime.timedelta(seconds=int(time.time() - start_time))}")

    # --- 5. Ridge Baseline ---
    print("\n[5/8] Training Ridge Baseline...")
    oof_ridge = np.full(len(X), np.nan, dtype=np.float64)
    test_ridge = np.zeros(len(X_test), dtype=np.float64)
    ridge_counts = np.zeros(len(X), dtype=np.int32)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_scaled)):
        model_ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])
        model_ridge.fit(X_scaled[tr_idx], y[tr_idx], sample_weight=recency_weights[tr_idx])
        preds = model_ridge.predict(X_scaled[va_idx])
        for i, vi in enumerate(va_idx):
            if np.isnan(oof_ridge[vi]):
                oof_ridge[vi] = 0.0
            oof_ridge[vi] += preds[i]
            ridge_counts[vi] += 1
        test_ridge += model_ridge.predict(X_test_scaled) / n_folds

    valid_ridge = ridge_counts > 0
    oof_ridge[valid_ridge] /= ridge_counts[valid_ridge]
    print(f"Ridge OOF R²: {r2_score(y[valid_ridge], oof_ridge[valid_ridge]):.5f}")

    # --- 6. Equal-Weight Ensemble ---
    print("\n[6/8] Equal-Weight Ensemble...")
    both = valid_mask & valid_ridge
    oof_blend = 0.5 * oof_mlp[both] + 0.5 * oof_ridge[both].astype(np.float32)
    print(f"Blend OOF R²: {r2_score(y[both], oof_blend):.5f}")
    test_blend = 0.5 * test_mlp + 0.5 * test_ridge.astype(np.float32)

    # --- 7. Shrinkage ---
    print("\n[7/8] Tuning shrinkage α...")
    best_r2, best_alpha = -np.inf, 1.0
    for alpha in np.linspace(0.3, 1.0, 71):
        score = r2_score(y[both], oof_blend * alpha)
        if score > best_r2:
            best_r2, best_alpha = score, alpha
    print(f"Optimal α = {best_alpha:.3f} → R² = {best_r2:.5f}")
    test_final = test_blend * best_alpha

    # --- 8. Save Output ---
    print("\n[8/8] Saving predictions...")
    os.makedirs(os.path.dirname(cfg["output_path"]) or '.', exist_ok=True)
    id_label = id_col if id_col else "index"
    df_sub = pd.DataFrame({id_label: test_ids, cfg["target_col"]: test_final.astype(np.float32)})
    df_sub.to_csv(cfg["output_path"], index=False)
    print(f"Saved → {cfg['output_path']}")
    print(f"Stats: mean={df_sub[cfg['target_col']].mean():.6f}, std={df_sub[cfg['target_col']].std():.6f}")
    print("🏁 Done!")


if __name__ == "__main__":
    main()
