import pandas as pd
import numpy as np
from tqdm import tqdm

def generate_and_save_data(n_samples, n_conventional_features, n_qdss_features, filepath):
    """
    Generates the synthetic dataset as described in the paper and saves it to a CSV file.
    This version uses a more realistic and structured approach to signal generation.
    """
    time = np.arange(n_samples)

    # --- Generate Conventional Sensor Data ---
    conventional_data = np.zeros((n_samples, n_conventional_features))
    for i in range(n_conventional_features // 2):
        freq = 24 * (i + 1)
        conventional_data[:, i*2] = np.sin(2 * np.pi * time / freq) + np.random.normal(0, 0.15, n_samples)
        conventional_data[:, i*2+1] = np.cos(2 * np.pi * time / freq) + np.random.normal(0, 0.15, n_samples)

    # --- Inject Stress Events ---
    stress_class = np.zeros(n_samples, dtype=int)
    yield_gt = np.full(n_samples, 10.0)
    num_events = 25
    stress_onsets = np.random.choice(np.arange(1000, n_samples - 1000), size=num_events, replace=False)
    stress_types = ([1, 2, 3] * (num_events // 3 + 1))[:num_events] # Drought, Pest, Nutrient Def.

    for i, onset in enumerate(stress_onsets):
        duration = np.random.randint(400, 900)
        end = min(onset + duration, n_samples)
        stress_class[onset:end] = stress_types[i]
        yield_reduction = np.linspace(0, 4.0 - stress_types[i], end - onset)
        yield_gt[onset:end] -= yield_reduction

    yield_gt += np.random.normal(0, 0.2, n_samples)

    # --- Generate QDSS Sensor Data based on stress class ---
    qdss_data = np.random.normal(0, 0.15, (n_samples, n_qdss_features)) # Base sensor noise
    for i in tqdm(range(n_samples), desc="Generating Realistic QDSS Signals"):
        if stress_class[i] != 0:
            # General stress signal (common to all non-healthy states)
            qdss_data[i, :] += 0.25 * np.random.normal(1, 0.1)
            
            # Class-specific signal with overlapping features
            class_idx = stress_class[i]
            start_peak = (class_idx - 1) * 20
            end_peak = start_peak + 30
            qdss_data[i, start_peak:end_peak] += np.random.normal(0.6, 0.15, 30)

    # --- Assemble DataFrame ---
    conv_cols = [f'Conv_{i}' for i in range(n_conventional_features)]
    qdss_cols = [f'QDSS_{i}' for i in range(n_qdss_features)]
    df_conv = pd.DataFrame(conventional_data, columns=conv_cols)
    df_qdss = pd.DataFrame(qdss_data, columns=qdss_cols)
    final_df = pd.concat([df_conv, df_qdss], axis=1)
    final_df['stress_class'] = stress_class
    final_df['yield'] = np.clip(yield_gt, 2, 10).astype(np.float32)

    # Save to file
    final_df.to_csv(filepath, index=False)