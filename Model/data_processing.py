import pandas as pd

def engineer_features(df, config):
    """Applies temporal feature engineering using rolling windows."""
    
    # Identify all sensor columns dynamically
    conv_cols = [f'Conv_{i}' for i in range(config["n_conventional_features"])]
    qdss_cols = [f'QDSS_{i}' for i in range(config["n_qdss_features"])]
    sensor_cols = conv_cols + qdss_cols
    
    new_features_df = df.copy()
    
    for window in [6, 12, 24]:
        for col in sensor_cols:
            new_features_df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
            new_features_df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()

    # Fill any initial NaN values from rolling operations
    new_features_df = new_features_df.fillna(method='bfill').fillna(method='ffill')
    return new_features_df