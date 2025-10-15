import os
import pandas as pd
# --- CHANGE 1: Direct imports instead of package imports ---
from data_simulation import generate_and_save_data
from training import run_full_ablation_study
from plotting import generate_all_plots
from utils import set_seeds, save_summary_file

# --- CENTRAL CONFIGURATION ---
CONFIG = {
    "n_samples": 20000,
    "n_clients": 20,
    "test_split_ratio": 0.2,
    "n_conventional_features": 12,
    "n_qdss_features": 96,
    # --- CHANGE 2: Paths adjusted to point to parent directory ---
    "data_path": "../data/synthetic_qal_dataset.csv",
    "results_dir": "../results/",
    "random_seed": 42, # Enter your seed for reproducibility
    "federated_rounds": 40,
    "local_epochs": 5,
    "fspm_hyperparams": {
        'lr': 0.00037,
        'dropout_rate': 0.5,
        'layer1_neurons': 192,
        'layer2_neurons': 112,
        'batch_size': 64,
    },
    "stress_classes": ['Healthy', 'Drought', 'Pest', 'Nutrient_Def']
}

def main():
    """Main function to run the entire QAL simulation pipeline."""
    print("--- Starting QAL Framework Simulation ---")

    # Create necessary directories
    os.makedirs(os.path.dirname(CONFIG["data_path"]), exist_ok=True)
    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    # Set random seeds for reproducibility
    set_seeds(CONFIG["random_seed"])
    print(f"Set random seed to {CONFIG['random_seed']} for reproducibility.")

    # --- Step 1: Data Generation ---
    if os.path.exists(CONFIG["data_path"]):
        print(f"Found existing sample dataset at {CONFIG['data_path']}.")
    else:
        print(f"Generating synthetic data at {CONFIG['data_path']}...")
        generate_and_save_data(
            n_samples=CONFIG["n_samples"],
            n_conventional_features=CONFIG["n_conventional_features"],
            n_qdss_features=CONFIG["n_qdss_features"],
            filepath=CONFIG["data_path"]
        )
        print("Data generation complete.")

    # --- Step 2: Run the full federated learning and ablation study ---
    print("\nRunning full ablation study... (This may take several minutes)")
    results = run_full_ablation_study(CONFIG)
    print("Ablation study complete.")

    # --- Step 3: Generate and save all result plots ---
    print(f"\nGenerating and saving plots to {CONFIG['results_dir']}...")
    generate_all_plots(results, CONFIG)
    print("Plot generation complete.")

    # --- Step 4: Save final numerical summary ---
    print(f"Saving numerical summary to {CONFIG['results_dir']}results-summary.txt...")
    save_summary_file(results, CONFIG)

    print("\n--- Simulation Finished Successfully ---")
    print(f"All outputs saved in the '{CONFIG['results_dir']}' directory.")


if __name__ == "__main__":
    main()