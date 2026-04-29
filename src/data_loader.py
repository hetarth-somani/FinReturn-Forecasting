import os
import random
import numpy as np
import pandas as pd
import gc

# 0.2 Set reproducibility seeds
SEED = 42

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(SEED)

def reduce_mem_usage(df):
    """Iterate through all columns of a dataframe and modify the data type to reduce memory usage."""
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage before: {start_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col != 'ID' and col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32) # float16 is tricky, use float32
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after: {end_mem:.2f} MB")
    return df

LOCAL = True
if LOCAL:
    DATA_DIR = 'dataset'
    TRAIN_FILE = 'train-001.parquet'
    TEST_FILE = 'test.parquet'
    SUB_FILE = 'sample_submission.csv'
else:
    DATA_DIR = '/kaggle/input/<dataset-name>'
    TRAIN_FILE = 'train.parquet'
    TEST_FILE = 'test.parquet'
    SUB_FILE = 'sample_submission.csv'

def main():
    print("--- Phase 0: Environment & Data Loading ---")
    
    # 0.3 Load data
    print(f"Loading train from {os.path.join(DATA_DIR, TRAIN_FILE)}...")
    df_train = pd.read_parquet(os.path.join(DATA_DIR, TRAIN_FILE))
    df_train = reduce_mem_usage(df_train)
    
    print(f"\nLoading test from {os.path.join(DATA_DIR, TEST_FILE)}...")
    df_test = pd.read_parquet(os.path.join(DATA_DIR, TEST_FILE))
    df_test = reduce_mem_usage(df_test)
    
    print(f"\nLoading submission from {os.path.join(DATA_DIR, SUB_FILE)}...")
    df_sub = pd.read_csv(os.path.join(DATA_DIR, SUB_FILE))
    
    # 0.4 Print shapes, dtypes
    print("\n--- Summary ---")
    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")
    print(f"Submission shape: {df_sub.shape}")
    
    # 0.5 Identify column groups
    id_col = 'ID'
    target_col = 'TARGET'
    price_col = 'Price'
    so3_t_col = 'SO3_T'
    
    all_features = [c for c in df_train.columns if c not in [id_col, target_col]]
    
    lag_t1_cols = [c for c in all_features if '_LagT1' in c]
    lag_t2_cols = [c for c in all_features if '_LagT2' in c]
    lag_t3_cols = [c for c in all_features if '_LagT3' in c]
    
    # Raw columns are features that are NOT lags and NOT special covariates like SO3_T (usually handled separate or part of raw)
    # Let's count them according to plan (112 raw)
    # Raw features don't have '_LagT' in their name
    raw_cols = [c for c in all_features if '_LagT' not in c and c not in [price_col, so3_t_col]]
    # wait, the plan states 112 raw features including Price and SO3_T. So if those are 2, there are 110 others. 
    # Let's just say raw features are all that don't have Lag suffix
    raw_features_all = [c for c in all_features if '_LagT' not in c]
    
    print(f"\nIdentified Columns:")
    print(f"  ID column: {id_col}")
    print(f"  TARGET column: {target_col}")
    print(f"  Price column exists: {price_col in df_train.columns}")
    print(f"  SO3_T column exists: {so3_t_col in df_train.columns}")
    print(f"  Total raw features (no lags): {len(raw_features_all)}")
    print(f"  Lag-T1 features: {len(lag_t1_cols)}")
    print(f"  Lag-T2 features: {len(lag_t2_cols)}")
    print(f"  Lag-T3 features: {len(lag_t3_cols)}")
    
    # 0.6 Verify Price lag columns
    price_lags = [c for c in all_features if 'Price' in c and '_LagT' in c]
    print(f"\nPrice lag columns found: {price_lags}")
    
    assert 'Price_LagT1' in df_train.columns, "Price_LagT1 missing!"
    assert 'Price_LagT2' in df_train.columns, "Price_LagT2 missing!"
    assert 'Price_LagT3' in df_train.columns, "Price_LagT3 missing!"
    
    print("\nPhase 0 check passed! 🎉")

if __name__ == "__main__":
    main()
