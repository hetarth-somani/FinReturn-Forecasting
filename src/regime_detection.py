import os
import gc
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

SEED = 42

def seed_everything(seed=42):
    np.random.seed(seed)
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    seed_everything(SEED)
    DATA_DIR = 'dataset'
    MODELS_DIR = 'models'
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    print("--- Phase 3: Stage 2 Regime Detection ---")
    
    print("Loading augmented datasets...")
    df_train = pd.read_parquet(os.path.join(DATA_DIR, 'train_stage1.parquet'))
    df_test = pd.read_parquet(os.path.join(DATA_DIR, 'test_stage1.parquet'))
    
    n_train = len(df_train)
    
    # 3.1 Select clustering features
    # Exclude: raw state features, ID, Price, TARGET, SO3_T
    # Include: all lag features + 13 price trajectory + cross-variable aggregates
    all_cols = df_train.columns.tolist()
    
    # Lag columns
    lag_cols = [c for c in all_cols if '_LagT' in c]
    
    # Stage 1 cross-variable features
    agg_cols = [c for c in all_cols if c.endswith('_momentum_score') or 
                                       c.endswith('_acceleration') or 
                                       c.endswith('_direction_consistency')]
    
    # Stage 1 price trajectory features
    price_traj_cols = ['price_momentum', 'price_curvature', 'price_trend_r2', 'price_volatility',
                       'price_range', 'price_return_t1', 'price_return_t2', 'price_return_t3',
                       'momentum_consistency', 'price_position', 'hurst_approx', 'sharpe_proxy', 'drawdown']
                       
    cluster_features = lag_cols + agg_cols + price_traj_cols
    print(f"Selected {len(cluster_features)} features for clustering.")
    
    # Combine train and test for clustering (no target leakage)
    print("Combining train and test for unsupervised clustering pipeline...")
    X_cluster_train = df_train[cluster_features]
    X_cluster_test = df_test[cluster_features]
    X_cluster_all = pd.concat([X_cluster_train, X_cluster_test], axis=0).reset_index(drop=True)
    
    # Free memory
    del X_cluster_train, X_cluster_test
    gc.collect()

    # 3.2 Build preprocessing pipeline
    print("\nBuilding clustering pipeline (Imputer -> RobustScaler -> PCA)...")
    clustering_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('pca', PCA(n_components=0.85, random_state=SEED))
    ])
    
    # 3.3 Fit pipeline
    print("Fitting PCA pipeline and transforming space... (this may take a minute)")
    X_pca = clustering_pipeline.fit_transform(X_cluster_all)
    print(f"PCA reduced dimensions to: {X_pca.shape[1]} components (explaining 85% variance)")
    
    # 3.4 Run K-Means for k=2..6
    print("\nEvaluating K-Means for k=2..6 ...")
    results = {}
    for k in [2, 3, 4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10, max_iter=300)
        labels = km.fit_predict(X_pca)
        
        # Sample for score to save time
        sil_score = silhouette_score(X_pca, labels, sample_size=10000, random_state=SEED)
        db_score = davies_bouldin_score(X_pca, labels)
        
        results[k] = {
            'silhouette': sil_score,
            'davies_bouldin': db_score,
            'model': km,
            'labels': labels
        }
        print(f"  k={k}: Silhouette={sil_score:.4f}, Davies-Bouldin={db_score:.4f}")
        
    # 3.5 Select optimal k
    # Highest silhouette score is primary metric
    best_k = max(results.keys(), key=lambda k: results[k]['silhouette'])
    print(f"\n=> Hand-picking Optimal K based on Silhouette: {best_k}")
    
    best_model = results[best_k]['model']
    best_labels = results[best_k]['labels']
    
    # Plot silhouette curve
    ks = list(results.keys())
    sils = [results[k]['silhouette'] for k in ks]
    plt.figure(figsize=(8, 5))
    plt.plot(ks, sils, marker='o', linestyle='-', color='b')
    plt.plot(best_k, results[best_k]['silhouette'], marker='*', color='r', markersize=15, label='Optimal k')
    plt.title("Silhouette Score vs. Number of Regimes (k)")
    plt.xlabel("Number of Regimes (k)")
    plt.ylabel("Silhouette Score (higher is better)")
    plt.xticks(ks)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join('eda_plots', 'kmeans_silhouette.png'))
    plt.close()

    # 3.6 Assign regime labels
    df_train['regime'] = best_labels[:n_train].astype(np.int8)
    df_test['regime'] = best_labels[n_train:].astype(np.int8)
    
    # 3.7 Compute per-regime statistics on training set
    print("\nRegime Validation (Train Set):")
    regimes = df_train['regime'].unique()
    valid_regimes = True
    
    stats_list = []
    target_data = {}
    for r in sorted(regimes):
        mask = df_train['regime'] == r
        subset = df_train[mask]
        
        cnt = mask.sum()
        pct = cnt / n_train * 100
        
        m_targ = subset['TARGET'].mean()
        s_targ = subset['TARGET'].std()
        
        m_mom = subset['price_momentum'].mean()
        m_vol = subset['price_volatility'].mean()
        m_hurst = subset['hurst_approx'].mean()
        
        target_data[r] = subset['TARGET'].dropna().values
        
        stats_list.append({
            'Regime': r,
            'Count': cnt,
            'Percent': pct,
            'TARGET_mean': m_targ,
            'TARGET_std': s_targ,
            'Mom_mean': m_mom,
            'Vol_mean': m_vol,
            'Hurst_mean': m_hurst
        })
        
        if pct < 5.0:
            print(f"  WARNING: Regime {r} has < 5% of data ({pct:.2f}%)!")
            valid_regimes = False
            
    stats_df = pd.DataFrame(stats_list)
    print(stats_df.to_string(index=False))
    
    # 3.8 Validate target distributions differ via KS test
    if best_k > 1:
        print("\nKS Test for TARGET distributions between regimes:")
        rs = sorted(target_data.keys())
        for i in range(len(rs)):
            for j in range(i+1, len(rs)):
                stat, p_val = ks_2samp(target_data[rs[i]], target_data[rs[j]])
                print(f"  Regime {rs[i]} vs {rs[j]} -> KS stat={stat:.4f}, p={p_val:.4e}")
                if p_val > 0.05:
                    print(f"  NOTE: Regimes {rs[i]} and {rs[j]} have statistically similar TARGET distributions.")
                    valid_regimes = False

    # 3.9 Save final results
    print("\nSaving clustering pipeline and datasets...")
    with open(os.path.join(MODELS_DIR, 'clustering_pipeline.pkl'), 'wb') as f:
        pickle.dump(clustering_pipeline, f)
        
    with open(os.path.join(MODELS_DIR, 'kmeans_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
        
    out_train = os.path.join(DATA_DIR, 'train_stage2.parquet')
    out_test = os.path.join(DATA_DIR, 'test_stage2.parquet')
    
    df_train.to_parquet(out_train, index=False)
    df_test.to_parquet(out_test, index=False)
    
    print(f"Train Dataset Saved: {out_train} (shape: {df_train.shape})")
    print(f"Test Dataset Saved: {out_test} (shape: {df_test.shape})")
    
    if not valid_regimes:
        print("\nWARNING: Some validation checks failed (e.g. regime size < 5% or similar TARGET dists).")
        print("You may want to manually reduce K in subsequent iterations if downstream models overfit.")
        
    print("\nPhase 3 Regime Detection complete! 🎉")

if __name__ == "__main__":
    main()
