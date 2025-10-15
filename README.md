# The Quantum-Enhanced Agri-Ledger (QAL)

This repository contains the official source code for the simulation experiments presented in the paper **"The Quantum-Enhanced Agri-Ledger (QAL): A Framework for Preemptive Sensing and Verifiable Sustainability"**.

The code is structured to be fully reproducible. Running the main script will either use the provided sample dataset or generate a new one, run the complete federated learning simulation and ablation study, and save all result plots and a numerical summary to the `results/` directory.

---

## 🚀 How to Run

Follow these steps to set up the environment and reproduce the paper's results.

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd QAL-Data
    ```

2.  **Create a Virtual Environment** (Highly Recommended)
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install Dependencies**
    Install all the required packages using the `requirements.txt` file from the root directory.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Simulation**
    Navigate into the `Model` directory and execute the main script.
    ```bash
    cd Model
    python main.py
    ```
    The script will print its progress to the console. The full simulation may take several minutes to complete depending on your hardware.

---

## 📂 Project Structure

The project is organized with all source code located within the `Model` directory.

```
.
├── Model/                  \# 🐍 The main source code package
│   ├── main.py             \# ✅ Main executable script to run everything
│   ├── data\_simulation.py  \# 🧬 Generates the synthetic dataset
│   ├── data\_processing.py  \# 🛠️ Handles feature engineering
│   ├── models.py           \# 🧠 Defines the PyTorch FSPM model
│   ├── training.py         \# 🏋️ Contains the main training logic
│   ├── plotting.py         \# 🎨 Functions to generate all plots
│   └── utils.py            \# ⚙️ Helper functions (Seeding)
│
├── Data/
│   └── synthetic\_qal\_dataset.csv  \# 📊 A sample dataset
│
├── results/                \# 📈 Output directory for plots (auto-generated)
│
├── README.md               \# 📄 This file
├── requirements.txt        \# 📦 List of required Python packages
```

-----

## 📊 Sample Dataset

The `Data/` directory contains a file named `synthetic_qal_dataset.csv`. This is a **sample dataset** generated from a single run of the simulation using the script `qal_package/data_simulation.py`.

  * When you run `main.py`, the script will **first check for this file**. If it exists, it will be used for the simulation to ensure quick and consistent runs.
  * If the `Data/` directory is empty or the file is missing, the script will **automatically generate a new dataset** before proceeding with the training.

-----

## 🖥️ Expected Output

After the `main.py` script finishes, the `results/` directory will be created and populated with the following files, which correspond to the figures in the paper:

  * `ablation_study.png`
  * `yield_comparison.png`
  * `confusion_matrix.png`
  * `federated_convergence.png`
  * `roc_curves.png`
  * `results-summary.txt` (A text file with the final RMSE, accuracy, and detailed ablation study numbers)


  Got it. That's a good change for keeping all the source code neatly in one place.

This new structure requires two small but important updates in `main.py` to make sure the imports and file paths work correctly from inside the `Model` directory.

Here are the updated files reflecting your new tree structure.

### **1. `main.py` (Updated)**

I've changed the `import` statements to be direct (since all files are in the same folder) and adjusted the file paths for the `data` and `results` directories to point to the parent folder.

```python
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
    "data_path": "../Data/synthetic_qal_dataset.csv",
    "results_dir": "../results/",
    "random_seed": 1, # Enter your seed for reproducibility
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
```

# Code for: The Quantum-Enhanced Agri-Ledger (QAL)

This repository contains the official source code for the simulation experiments presented in the paper **"The Quantum-Enhanced Agri-Ledger (QAL): A Framework for Preemptive Sensing and Verifiable Sustainability"**.

The code is structured to be fully reproducible. Running the main script will either use the provided sample dataset or generate a new one, run the complete federated learning simulation and ablation study, and save all result plots and a numerical summary to the `results/` directory.

---

## 🚀 How to Run

Follow these steps to set up the environment and reproduce the paper's results.

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd QAL-Data
    ```

2.  **Create a Virtual Environment** (Highly Recommended)
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install Dependencies**
    Install all the required packages using the `requirements.txt` file from the root directory.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Simulation**
    Navigate into the `Model` directory and execute the main script.
    ```bash
    cd Model
    python main.py
    ```
    The script will print its progress to the console. The full simulation may take several minutes to complete depending on your hardware.

---

## 📂 Project Structure

The project is organized with all source code located within the `Model` directory.

````



```
```