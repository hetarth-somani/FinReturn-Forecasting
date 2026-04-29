import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

SEED = 42
def seed_everything(seed=42):
    np.random.seed(seed)
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    seed_everything(SEED)
    
    # Configuration
    DATA_DIR = 'dataset'
    TRAIN_FILE = 'train-001.parquet'
    OUT_DIR = 'eda_plots'
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        
    print("Loading training data for EDA...")
    df = pd.read_parquet(os.path.join(DATA_DIR, TRAIN_FILE))
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # 1.1 Compute missing value percentages
    print("\n[1.1] Missing Values Analysis...")
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    print(f"Features with missing values: {len(missing)}")
    if len(missing) > 0:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=missing.head(20).index, y=missing.head(20).values)
        plt.xticks(rotation=90)
        plt.title("Top 20 Features with Missing Values (%)")
        plt.ylabel("% Missing")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'missing_values.png'))
        plt.close()
    
    # 1.2 TARGET distribution
    print("\n[1.2] TARGET Distribution Analysis...")
    target = df['TARGET'].dropna()
    print(f"Mean: {target.mean():.6f}, Std: {target.std():.6f}")
    print(f"Skew: {target.skew():.6f}, Kurtosis: {target.kurtosis():.6f}")
    print(f"Min: {target.min():.6f}, Max: {target.max():.6f}")
    print("Percentiles:")
    for p in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]:
        print(f" {p*100:2.0f}%: {target.quantile(p):.6f}")
        
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(target, bins=100, kde=True, ax=axes[0])
    axes[0].set_title("TARGET Histogram")
    sns.boxplot(x=target, ax=axes[1])
    axes[1].set_title("TARGET Boxplot")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'target_dist.png'))
    plt.close()
    
    # 1.3 Price distribution
    print("\n[1.3] Price Distribution Analysis...")
    if 'Price' in df.columns:
        price = df['Price'].dropna()
        Q1, Q3 = price.quantile(0.25), price.quantile(0.75)
        IQR = Q3 - Q1
        outliers = price[(price < (Q1 - 1.5 * IQR)) | (price > (Q3 + 1.5 * IQR))]
        print(f"Price Outliers (1.5 IQR): {len(outliers)} / {len(price)} ({len(outliers)/len(price)*100:.2f}%)")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.histplot(price, bins=100, kde=True, ax=axes[0])
        axes[0].set_title("Price Histogram")
        sns.boxplot(x=price, ax=axes[1])
        axes[1].set_title("Price Boxplot")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'price_dist.png'))
        plt.close()
        
    # 1.4 Correlation of lag features with TARGET
    print("\n[1.4] Lag Features Correlation with TARGET...")
    lag_cols = [c for c in df.columns if '_LagT' in c]
    # Sample data for speed if very large
    sample_df = df.sample(n=min(100000, len(df)), random_state=SEED) if len(df) > 100000 else df
    corrs = []
    
    # Compute correlation efficiently
    target_sample = sample_df['TARGET']
    for col in lag_cols:
        col_sample = sample_df[col]
        valid_idx = ~(col_sample.isna() | target_sample.isna())
        if valid_idx.sum() > 100:
            c, _ = stats.pearsonr(col_sample[valid_idx], target_sample[valid_idx])
            corrs.append({'feature': col, 'corr': c, 'abs_corr': abs(c)})
            
    if corrs:
        corr_df = pd.DataFrame(corrs).sort_values(by='abs_corr', ascending=False)
        print("Top 10 correlated lag features:")
        print(corr_df.head(10)[['feature', 'corr']])
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=corr_df.head(20), x='corr', y='feature')
        plt.title("Top 20 Lag Features Correlated with TARGET")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'top_correlations.png'))
        plt.close()
    
    # 1.5 Identify near-constant features
    print("\n[1.5] Inspecting near-constant features (std < 1e-6)...")
    stds = sample_df.std(numeric_only=True)
    constant_features = stds[stds < 1e-6].index.tolist()
    print(f"Found {len(constant_features)} near-constant features")
    if len(constant_features) > 0:
        print(f"First 5: {constant_features[:5]}")
        
    # 1.6 Visualize Price trajectory for 5 sample rows
    print("\n[1.6] Visualizing Price trajectories...")
    if all(c in df.columns for c in ['Price', 'Price_LagT1', 'Price_LagT2', 'Price_LagT3']):
        plt.figure(figsize=(10, 6))
        sample_rows = df.sample(n=5, random_state=SEED)
        
        p3 = sample_rows['Price'].values
        p2 = sample_rows['Price'].values - sample_rows['Price_LagT1'].values
        p1 = sample_rows['Price'].values - sample_rows['Price_LagT2'].values
        p0 = sample_rows['Price'].values - sample_rows['Price_LagT3'].values
        
        for i in range(5):
            traj = [p0[i], p1[i], p2[i], p3[i]]
            plt.plot([0, 1, 2, 3], traj, marker='o', label=f'Row {sample_rows.index[i]}')
        
        plt.xticks([0, 1, 2, 3], ['t-T3', 't-T2', 't-T1', 't'])
        plt.title("Sample Price Trajectories")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUT_DIR, 'sample_trajectories.png'))
        plt.close()
        
    # 1.7 Examine SO3_T distribution
    print("\n[1.7] SO3_T Analysis...")
    if 'SO3_T' in df.columns:
        so3t_sample = sample_df['SO3_T']
        valid_idx = ~(so3t_sample.isna() | target_sample.isna())
        if valid_idx.sum() > 2:
            so3_corr, _ = stats.pearsonr(so3t_sample[valid_idx], target_sample[valid_idx])
            print(f"SO3_T correlation with TARGET: {so3_corr:.6f}")
        
        plt.figure(figsize=(10, 5))
        sns.histplot(df['SO3_T'].dropna(), bins=50, kde=True)
        title_text = f"SO3_T Distribution (corr with TARGET: {so3_corr:.4f})" if 'so3_corr' in locals() else "SO3_T Distribution"
        plt.title(title_text)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'so3_t_analysis.png'))
        plt.close()

    # 1.8 Group features by prefix family
    print("\n[1.8] Feature Family Grouping...")
    # e.g., 'S01_F01_U01' -> 'S01'
    base_features = [c for c in df.columns if '_LagT' not in c and c not in ['ID', 'TARGET', 'Price', 'SO3_T']]
    families = {}
    for f in base_features:
        parts = f.split('_')
        if len(parts) > 0:
            fam = parts[0]
            if fam not in families:
                families[fam] = 0
            families[fam] += 1
            
    print("Feature counts by primary family prefix:")
    for fam, cnt in sorted(families.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {fam}: {cnt} features")
        
    print("\nEDA checks completed. Plots saved to 'eda_plots/' directory. 🎉")

if __name__ == "__main__":
    main()
