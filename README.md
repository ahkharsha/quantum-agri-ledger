# The Quantum-Enhanced Agri-Ledger (QAL)

This repository contains the official source code for the simulation experiments presented in the paper **"The Quantum-Enhanced Agri-Ledger: A Simulation-Based Pathway to Incentivized Climate-Smart Agronomy"**.

The implementation has been consolidated into a **single, self-contained Jupyter Notebook** to ensure ease of review and absolute reproducibility. Running this notebook will:

1. Generate the synthetic dataset (QDSS + Conventional sensors).
2. Train the Federated Smart Prediction Model (FSPM) over 40 communication rounds.
3. Conduct a comparative ablation study against baseline models.
4. Automatically generate all figures and package the results into a submission-ready archive.

---

## 🚀 How to Run

### Option 1: Local Execution (Recommended)

1. **Clone the Repository**
```bash
git clone <your-repository-url>
cd QAL-Data

```


2. **Install Dependencies**
Ensure you have Python 3.8+ and Jupyter installed.
```bash
pip install notebook torch torchvision pandas numpy scikit-learn matplotlib seaborn simpy tqdm

```


3. **Launch the Notebook**
```bash
jupyter notebook QAL_Implementation.ipynb

```


4. **Reproduce Results**
* In the Jupyter interface, go to **Kernel** > **Restart & Run All**.
* The simulation will take approximately 10–20 minutes depending on your GPU/CPU.



### Option 2: Google Colab

1. Upload the `.ipynb` file to Google Drive.
2. Open it with Google Colab.
3. Change the runtime type to **GPU** (Runtime > Change runtime type > T4 GPU).
4. Select **Runtime** > **Run all**.

---

## 📂 Repository Structure

```text
.
├── QAL_Implementation.ipynb    # 📓 Main experiment notebook (Run this)
├── QAL_Experiment_Results/     # 📂 Output folder (Created automatically)
│   ├── federated_learning_convergence.png
│   ├── mean_confusion_matrix.png
│   ├── mean_roc_curves.png
│   ├── noise_sensitivity_analysis.png
│   ├── security_stress_test.png
│   ├── qal_fspm_model.pth      # 🧠 Saved Model Weights
│   └── feature_scaler.pkl      # ⚖️ Feature Scaler
│
├── qal_results.zip             # 📦 Zipped artifacts (Ready for submission)
└── README.md                   # 📄 This file

```

---

## 📊 Outputs & Artifacts

Upon successful completion, the notebook generates two primary outputs in your working directory:

### 1. `QAL_Experiment_Results/` (Folder)

This directory contains high-resolution `.png` figures corresponding to the paper:

* **Federated Convergence:** Visualizes global model accuracy over 40 communication rounds.
* **Confusion Matrix:** Mean classification performance across "Healthy," "Drought," "Pest," and "Nutrient Def."
* **ROC Curves:** Sensitivity vs. Specificity analysis for all stress classes.
* **Ablation Study:** Bar charts comparing QAL against Random Forest and SVM baselines.
* **Robustness Analysis:** Performance metrics under varying noise levels ().

### 2. `qal_results.zip` (Archive)

A compressed archive containing the full contents of the results folder. This file is intended to be attached as **Supplementary Material** for peer review, allowing reviewers to verify the model weights (`qal_fspm_model.pth`) and scaler (`feature_scaler.pkl`) without needing to re-run the full training simulation.