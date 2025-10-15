import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import os

def plot_ablation_study(results, save_path):
    """Generates and saves the ablation study bar chart."""
    df = pd.DataFrame(results['ablation_results'])

    fig, ax1 = plt.subplots(figsize=(10, 7))

    color_rmse = 'darkblue'
    color_acc = 'darkred'
    
    x = np.arange(len(df['Model']))
    width = 0.4
    
    rects1 = ax1.bar(x - width/2, df['RMSE (tons/ha)'], width, label='Yield RMSE', color=color_rmse)
    ax1.set_xlabel('Model Configuration', fontsize=14, labelpad=15)
    ax1.set_ylabel('Yield Prediction RMSE (tons/ha)', color=color_rmse, fontsize=14)
    ax1.tick_params(axis='y', labelcolor=color_rmse)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Model'], rotation=15, ha="right")

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, df['Accuracy'], width, label='Stress Accuracy', color=color_acc)
    ax2.set_ylabel('Stress Classification Accuracy', color=color_acc, fontsize=14)
    ax2.tick_params(axis='y', labelcolor=color_acc)
    ax2.set_ylim(0, 1.05)

    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', color=color_rmse)
    for rect in rects2:
        height = rect.get_height()
        ax2.annotate(f'{height:.1%}', xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', color=color_acc)

    fig.tight_layout()
    plt.title('Ablation Study: Model Performance Comparison', fontsize=16)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_yield_comparison(results, save_path):
    """Generates and saves the yield prediction scatter plot."""
    preds = results['predictions']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # Plot a random sample to avoid overplotting
    sample_indices = np.random.choice(len(preds['y_true_yield']), size=200, replace=False)
    y_true_sample = np.array(preds['y_true_yield'])[sample_indices]
    y_pred_baseline_sample = np.array(preds['y_pred_yield_baseline'])[sample_indices]
    y_pred_qal_sample = np.array(preds['y_pred_yield_qal'])[sample_indices]

    ax1.scatter(y_true_sample, y_pred_baseline_sample, alpha=0.6, c='crimson', edgecolors='k')
    ax1.plot([2, 10], [2, 10], 'k--')
    rmse_baseline = results['ablation_results'][0]['RMSE (tons/ha)']
    ax1.set_title(f'Baseline Model (RF on Conv Data)\nRMSE: {rmse_baseline:.2f} tons/ha', fontsize=14)
    ax1.set_xlabel('Actual Yield (tons/ha)', fontsize=12)
    ax1.set_ylabel('Predicted Yield (tons/ha)', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_aspect('equal', 'box')
    ax1.set_xlim(2, 10); ax1.set_ylim(2, 10)

    ax2.scatter(y_true_sample, y_pred_qal_sample, alpha=0.6, c='forestgreen', edgecolors='k')
    ax2.plot([2, 10], [2, 10], 'k--')
    rmse_qal = results['fspm_full']['rmse']
    ax2.set_title(f'Full QAL Model (FSPM)\nRMSE: {rmse_qal:.2f} tons/ha', fontsize=14)
    ax2.set_xlabel('Actual Yield (tons/ha)', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.set_aspect('equal', 'box')
    ax2.set_xlim(2, 10); ax2.set_ylim(2, 10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix(results, config, save_path):
    """Generates and saves the confusion matrix for the FSPM model."""
    cm = confusion_matrix(results['predictions']['y_true_stress'], results['predictions']['y_pred_stress_qal'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=config['stress_classes'],
                yticklabels=config['stress_classes'])
    plt.ylabel('Actual Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.title('Confusion Matrix for Full QAL Model (FSPM)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_federated_convergence(results, save_path):
    """Plots the model accuracy over federated learning rounds."""
    df = pd.DataFrame(results['convergence'])
    plt.figure(figsize=(10, 6))
    plt.plot(df['round'], df['accuracy'], marker='o', linestyle='-', color='m')
    plt.title('Federated Learning Convergence', fontsize=16)
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Global Model Accuracy', fontsize=12)
    plt.grid(True, linestyle=':')
    plt.ylim(bottom=max(0, df['accuracy'].min() - 0.1), top=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curves(results, config, save_path):
    """Generates and saves ROC curves for the FSPM model."""
    y_true_one_hot = pd.get_dummies(results['predictions']['y_true_stress']).values
    y_pred_probs = results['predictions']['y_pred_stress_probs_qal']
    
    plt.figure(figsize=(10, 8))
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'darkgreen']
    for i, (class_name, color) in enumerate(zip(config['stress_classes'], colors)):
        fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC of class {class_name} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def generate_all_plots(results, config):
    """A wrapper function to call all plotting functions."""
    res_dir = config['results_dir']
    plot_ablation_study(results, os.path.join(res_dir, "ablation_study.png"))
    plot_yield_comparison(results, os.path.join(res_dir, "yield_comparison.png"))
    plot_confusion_matrix(results, config, os.path.join(res_dir, "confusion_matrix.png"))
    plot_federated_convergence(results, os.path.join(res_dir, "federated_convergence.png"))
    plot_roc_curves(results, config, os.path.join(res_dir, "roc_curves.png"))