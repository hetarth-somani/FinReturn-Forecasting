import os
import gc
import numpy as np
import pandas as pd

def stage1_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  DataFrame with raw features + lag features + Price
    Output: DataFrame with additional engineered columns appended
    """
    print(f"Original shape: {df.shape}")
    
    # ---------------------------------------------------------
    # PART 1: Price Trajectory Reconstruction
    # ---------------------------------------------------------
    # Note: T3 > T2 > T1 so lagT3 is the oldest price
    p3 = df['Price'].values.copy()                       # Current
    p2 = p3 - df['Price_LagT1'].values                   # Most recent lag
    p1 = p3 - df['Price_LagT2'].values                   # Middle lag
    p0 = p3 - df['Price_LagT3'].values                   # Oldest lag
    
    # NaN handling
    # If Price or any lag is NaN, fill the entire trajectory with 0s to prevent math errors downstream
    invalid = np.isnan(p3) | np.isnan(p2) | np.isnan(p1) | np.isnan(p0)
    p3[invalid] = 0
    p2[invalid] = 0
    p1[invalid] = 0
    p0[invalid] = 0
    
    prices = np.column_stack([p0, p1, p2, p3]) # shape (n, 4)
    t = np.array([0, 1, 2, 3])
    t_mean = t.mean()
    p_mean = prices.mean(axis=1, keepdims=True)
    epsilon = 1e-10

    # ---------------------------------------------------------
    # PART 2: 13 Price Features
    # ---------------------------------------------------------
    
    # 1. price_momentum (linear slope)
    numerator = ((t - t_mean) * (prices - p_mean)).sum(axis=1)
    denominator = ((t - t_mean) ** 2).sum()
    df['price_momentum'] = (numerator / denominator).astype(np.float32)
    
    # 2. price_curvature (quadratic x^2 coefficient)
    # V = np.vander(t, 3) --> [[0,0,1], [1,1,1], [4,2,1], [9,3,1]]
    V = np.vander(t, 3)
    coeffs = np.linalg.lstsq(V, prices.T, rcond=None)[0]
    df['price_curvature'] = coeffs[0].astype(np.float32)
    
    # 3. price_trend_r2
    y_pred = df['price_momentum'].values[:, None] * t + p_mean
    ss_res = ((prices - y_pred) ** 2).sum(axis=1)
    ss_tot = ((prices - p_mean) ** 2).sum(axis=1)
    df['price_trend_r2'] = np.clip(1 - ss_res / (ss_tot + epsilon), 0, 1).astype(np.float32)
    
    # 4. price_volatility
    df['price_volatility'] = prices.std(axis=1, ddof=0).astype(np.float32)
    
    # 5. price_range
    df['price_range'] = (prices.max(axis=1) - prices.min(axis=1)).astype(np.float32)
    
    # 6-8. returns
    df['price_return_t1'] = ((p3 - p2) / (np.abs(p2) + epsilon) * 100).astype(np.float32)
    df['price_return_t2'] = ((p3 - p1) / (np.abs(p1) + epsilon) * 100).astype(np.float32)
    df['price_return_t3'] = ((p3 - p0) / (np.abs(p0) + epsilon) * 100).astype(np.float32)
    
    # 9. momentum_consistency (binary)
    d1 = np.sign(p1 - p0)
    d2 = np.sign(p2 - p1)
    d3 = np.sign(p3 - p2)
    df['momentum_consistency'] = ((d1 == d2) & (d2 == d3)).astype(np.int8)
    
    # 10. price_position
    df['price_position'] = ((p3 - p0) / (df['price_range'].values + epsilon)).astype(np.float32)
    
    # 11. hurst_approx
    diffs_1 = np.diff(prices, axis=1) # p1-p0, p2-p1, p3-p2
    var_1 = diffs_1.var(axis=1)
    diffs_2 = prices[:, 2:] - prices[:, :-2] # p2-p0, p3-p1
    var_2 = diffs_2.var(axis=1)
    
    hurst = 0.5 * np.log(var_2 + epsilon) / np.log(2) - 0.5 * np.log(var_1 + epsilon) / np.log(2)
    # Default to 0.5 if var_1 is essentially 0
    hurst[var_1 <= epsilon] = 0.5
    df['hurst_approx'] = np.clip(hurst, 0, 1).astype(np.float32)
    
    # 12. sharpe_proxy
    df['sharpe_proxy'] = (df['price_momentum'].values / (df['price_volatility'].values + epsilon)).astype(np.float32)
    
    # 13. drawdown
    peak = prices.max(axis=1)
    df['drawdown'] = ((p3 - peak) / (peak + epsilon) * 100).astype(np.float32)

    # ---------------------------------------------------------
    # PART 3: Cross-Variable Lag Aggregations
    # ---------------------------------------------------------
    lag_cols = [c for c in df.columns if '_LagT1' in c]
    base_features = [c.replace('_LagT1', '') for c in lag_cols]
    
    print(f"Found {len(base_features)} base features with lag variants.")
    
    for X in base_features:
        lt1 = df[f'{X}_LagT1'].values
        lt2 = df[f'{X}_LagT2'].values
        lt3 = df[f'{X}_LagT3'].values
        
        df[f'{X}_momentum_score'] = (0.5 * lt1 + 0.3 * lt2 + 0.2 * lt3).astype(np.float32)
        df[f'{X}_acceleration'] = (lt1 - lt2).astype(np.float32)
        df[f'{X}_direction_consistency'] = (np.sign(lt1) == np.sign(lt2)).astype(np.int8)
        
    print(f"New shape after Stage 1 Feature Engineering: {df.shape}")
    
    # Validate missing values in new engineered features
    new_price_features = ['price_momentum', 'price_curvature', 'price_trend_r2', 'price_volatility',
                          'price_range', 'price_return_t1', 'price_return_t2', 'price_return_t3',
                          'momentum_consistency', 'price_position', 'hurst_approx', 'sharpe_proxy', 'drawdown']
    
    nan_counts = df[new_price_features].isna().sum()
    if nan_counts.sum() > 0:
        print("WARNING: NaNs found in price features!")
        print(nan_counts[nan_counts > 0])
        
    return df

def main():
    DATA_DIR = 'dataset'
    
    # Process Train
    print("--- Processing Train Data ---")
    df_train = pd.read_parquet(os.path.join(DATA_DIR, 'train-001.parquet'))
    df_train = stage1_feature_engineering(df_train)
    
    out_train = os.path.join(DATA_DIR, 'train_stage1.parquet')
    print(f"Saving augmented train data to {out_train} ...")
    df_train.to_parquet(out_train, index=False)
    del df_train
    gc.collect()
    
    # Process Test
    print("\n--- Processing Test Data ---")
    df_test = pd.read_parquet(os.path.join(DATA_DIR, 'test.parquet'))
    df_test = stage1_feature_engineering(df_test)
    
    out_test = os.path.join(DATA_DIR, 'test_stage1.parquet')
    print(f"Saving augmented test data to {out_test} ...")
    df_test.to_parquet(out_test, index=False)
    del df_test
    gc.collect()
    
    print("\nPhase 2 Feature Engineering complete! 🎉")

if __name__ == "__main__":
    main()
