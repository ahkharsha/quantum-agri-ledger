# The Quantum-Enhanced Agri-Ledger (QAL) - Dataset

This repository contains the synthetic dataset used for the simulation experiments presented in the paper **"The Quantum-Enhanced Agri-Ledger: A Simulation-Based Pathway to Incentivized Climate-Smart Agronomy"**.

To ensure ease of access and focused review of the underlying data, this repository strictly hosts the generated experimental data file: `synthetic_qal_dataset.csv`.

---

## 📊 Dataset Overview

The dataset consists of synthesized Quantum Dot Sensor System (QDSS) and conventional agricultural sensor readings. It is structured to support both multi-class classification and continuous regression tasks.

**File Location:** `Dataset/synthetic_qal_dataset.csv`

### Data Dictionary & Features

The dataset contains the following feature categories:

* **Conventional Sensor Readings:** `temp`, `humidity`, `soil_moisture`.
* **Temporal & Time-Series Features:** Cyclic time encoding (`hour_sin`, `hour_cos`), lag variables (e.g., `temp_lag_6`), and rolling means to capture environmental trends.
* **Quantum Dot Sensor System (QDSS):** 96 distinct spectral reading columns (`qdss_0` to `qdss_95`) representing the high-resolution nanoscale optical signatures of the crops.
* **Targets:**
* `stress_label` / `stress_type`: Target for classification models (Healthy, Drought, Pest Infestation, Nutrient Deficiency).
* `yield`: A continuous target variable representing the projected crop yield for regression modeling.



---

## 🚀 How to Access

You can download the dataset directly from the repository or clone it locally:

```bash
git clone https://github.com/ahkharsha/quantum-agri-ledger.git
cd quantum-agri-ledger

```

---

## 🛠️ Usage

This `.csv` file can be imported directly into your preferred data analysis or machine learning environment (e.g., Python/Pandas, R, MATLAB) for independent validation of the data distributions, feature engineering, or custom model training.

Example loading the data in Python:

```python
import pandas as pd

# Load the QAL dataset
df = pd.read_csv('Dataset/synthetic_qal_dataset.csv')

# Separate features and target
X_qdss = df.loc[:, 'qdss_0':'qdss_95']
y_class = df['stress_label']
y_reg = df['yield']

print(df.info())

```