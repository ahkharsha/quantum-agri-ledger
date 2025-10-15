import torch
import numpy as np
import random
import pandas as pd
from sklearn.metrics import classification_report

def set_seeds(seed_value: int):
    """
    Sets the random seeds for torch, numpy, and python for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_summary_file(results, config):
    """Saves a final text file with all key numerical results."""
    summary_file_path = f"{config['results_dir']}/results-summary.txt"
    ablation_df = pd.DataFrame(results['ablation_results'])
    
    with open(summary_file_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("QAL FRAMEWORK - FINAL SIMULATION RESULTS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("--- Overall Performance (Full QAL Model) ---\n")
        f.write(f"Final Accuracy: {results['fspm_full']['accuracy']:.2%}\n")
        f.write(f"Final RMSE:     {results['fspm_full']['rmse']:.2f} tons/ha\n\n")
        
        f.write("\n--- Comprehensive Ablation Study Results ---\n")
        f.write(ablation_df.to_string(index=False))
        
        f.write("\n\n\n--- Detailed Classification Report for Full QAL Model ---\n")
        report = classification_report(
            results['predictions']['y_true_stress'], 
            results['predictions']['y_pred_stress_qal'],
            target_names=config['stress_classes']
        )
        f.write(report)
        
        f.write("\n\n" + "="*70)