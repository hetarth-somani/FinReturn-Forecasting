# ============================================
# Adversarial Validation
# ============================================
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import os

DATA_DIR = 'dataset'

# Load both datasets
print("Loading data...")
df_train = pd.read_parquet(os.path.join(DATA_DIR, 'train_stage1.parquet'))
df_test = pd.read_parquet(os.path.join(DATA_DIR, 'test_stage1.parquet'))

# Define features (same as model training)
drop_cols = ['ID', 'TARGET', 'Price']
features = [c for c in df_train.columns if c not in drop_cols]
# Also drop regime if present
features = [c for c in features if c != 'regime']

X_train = df_train[features].values
X_test = df_test[features].values

# Subsample for faster execution
SAMPLE_SIZE = 50000
np.random.seed(42)
if len(X_train) > SAMPLE_SIZE:
    idx_train = np.random.choice(len(X_train), SAMPLE_SIZE, replace=False)
    X_train = X_train[idx_train]
if len(X_test) > SAMPLE_SIZE:
    idx_test = np.random.choice(len(X_test), SAMPLE_SIZE, replace=False)
    X_test = X_test[idx_test]

# Create adversarial labels: 0 = train, 1 = test
y_adv = np.hstack([np.zeros(len(X_train)), np.ones(len(X_test))])
X_adv = np.vstack([X_train, X_test])

print(f"Combined dataset: {X_adv.shape}")
print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

# Train a classifier to distinguish train from test
clf = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    verbosity=-1
)

# 5-fold cross-validated AUC
print("\nRunning adversarial validation (5-fold CV)...")
auc_scores = cross_val_score(clf, X_adv, y_adv, cv=5, scoring='roc_auc')
print(f"\n{'='*50}")
print(f"Adversarial Validation AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
print(f"{'='*50}")

# Interpret
if auc_scores.mean() < 0.55:
    print("✅ Train ≈ Test distributions. Your CV is trustworthy!")
elif auc_scores.mean() < 0.70:
    print("⚠️ Mild distribution shift detected. CV may be slightly optimistic.")
elif auc_scores.mean() < 0.90:
    print("❌ Significant distribution shift! Your CV scores are NOT reliable.")
else:
    print("🚨 SEVERE shift! Train and test are fundamentally different.")

# Get feature importances (which features differ most)
print("\nTraining full model for feature importance analysis...")
clf.fit(X_adv, y_adv)
importances = sorted(
    zip(features, clf.feature_importances_),
    key=lambda x: x[1], reverse=True
)
print("\nTop 15 features causing train/test distribution shift:")
for name, imp in importances[:15]:
    print(f"  {name}: {imp}")
