import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from copy import deepcopy

from .models import FSPM_PyTorch
from .data_processing import engineer_features

def _client_update(model, optimizer, train_loader, epochs, device, config):
    """Performs local training on a client's data."""
    model.train()
    criterion_yield = nn.MSELoss()
    criterion_stress = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for data, yield_true, stress_true in train_loader:
            data, yield_true, stress_true = data.to(device), yield_true.to(device), stress_true.to(device)
            optimizer.zero_grad()
            yield_pred, stress_pred = model(data)
            loss_yield = criterion_yield(yield_pred.squeeze(), yield_true)
            loss_stress = criterion_stress(stress_pred, stress_true)
            loss = 0.5 * loss_yield + 0.5 * loss_stress
            loss.backward()
            optimizer.step()
    return model.state_dict()

def _server_aggregate(global_model, client_weights):
    """Averages the weights from clients to update the global model."""
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_weights[i][k].float() for i in range(len(client_weights))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

def _evaluate_fspm_model(model, test_loader, device):
    """Evaluates the FSPM model on the test set."""
    model.eval()
    all_yield_preds, all_yield_true = [], []
    all_stress_preds_labels, all_stress_true = [], []
    all_stress_preds_probs = []

    with torch.no_grad():
        for data, yield_true, stress_true in test_loader:
            data = data.to(device)
            yield_pred, stress_pred_logits = model(data)
            
            all_yield_preds.extend(yield_pred.squeeze().cpu().numpy())
            all_yield_true.extend(yield_true.numpy())
            
            stress_pred_labels = torch.argmax(stress_pred_logits, dim=1)
            all_stress_preds_labels.extend(stress_pred_labels.cpu().numpy())
            all_stress_true.extend(stress_true.numpy())
            all_stress_preds_probs.extend(torch.softmax(stress_pred_logits, dim=1).cpu().numpy())
            
    rmse = np.sqrt(mean_squared_error(all_yield_true, all_yield_preds))
    accuracy = accuracy_score(all_stress_true, all_stress_preds_labels)
    
    return rmse, accuracy, all_yield_true, all_yield_preds, all_stress_true, all_stress_preds_labels, np.array(all_stress_preds_probs)

def run_full_ablation_study(config):
    """
    Runs all model configurations (RF, SVM, FSPM) and returns a comprehensive results dictionary.
    """
    df = pd.read_csv(config['data_path'])
    
    # --- Data Splitting and Feature Engineering ---
    train_df, test_df = train_test_split(df, test_size=config['test_split_ratio'], shuffle=False, random_state=config['random_seed'])
    
    print("Engineering features for training and test sets...")
    train_featured = engineer_features(train_df, config)
    test_featured = engineer_features(test_df, config)

    # Define feature sets
    conv_features = [f'Conv_{i}' for i in range(config['n_conventional_features'])]
    qdss_features = [f'QDSS_{i}' for i in range(config['n_qdss_features'])]
    all_sensor_features = conv_features + qdss_features
    
    feature_cols = [col for col in train_featured.columns if col not in ['stress_class', 'yield']]
    
    # --- Data Scaling ---
    scaler = StandardScaler().fit(train_featured[feature_cols])
    X_train_scaled = scaler.transform(train_featured[feature_cols])
    X_test_scaled = scaler.transform(test_featured[feature_cols])
    
    y_yield_train = train_featured['yield'].values
    y_stress_train = train_featured['stress_class'].values
    y_yield_test = test_featured['yield'].values
    y_stress_test = test_featured['stress_class'].values

    # Get indices for conventional-only features
    conv_feature_indices = [feature_cols.index(f) for f in feature_cols if not f.startswith('QDSS_')]

    X_train_conv_scaled = X_train_scaled[:, conv_feature_indices]
    X_test_conv_scaled = X_test_scaled[:, conv_feature_indices]

    ablation_results = []
    
    # --- Baseline 1: Conventional Data + Random Forest ---
    print("Training Baseline: Conventional + Random Forest...")
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=config['random_seed'], n_jobs=-1).fit(X_train_conv_scaled, y_yield_train)
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=config['random_seed'], n_jobs=-1).fit(X_train_conv_scaled, y_stress_train)
    y_pred_yield_rf_conv = rf_reg.predict(X_test_conv_scaled)
    y_pred_stress_rf_conv = rf_clf.predict(X_test_conv_scaled)
    ablation_results.append({
        'Model': 'Conventional Data + RF',
        'RMSE (tons/ha)': np.sqrt(mean_squared_error(y_yield_test, y_pred_yield_rf_conv)),
        'Accuracy': accuracy_score(y_stress_test, y_pred_stress_rf_conv)
    })

    # --- Baseline 2: Conventional Data + SVM ---
    print("Training Baseline: Conventional + SVM...")
    # Use a subset for SVM training to speed it up
    subset_indices = np.random.choice(X_train_conv_scaled.shape[0], 2000, replace=False)
    svr = SVR().fit(X_train_conv_scaled[subset_indices], y_yield_train[subset_indices])
    svc = SVC(probability=True).fit(X_train_conv_scaled[subset_indices], y_stress_train[subset_indices])
    y_pred_yield_svm_conv = svr.predict(X_test_conv_scaled)
    y_pred_stress_svm_conv = svc.predict(X_test_conv_scaled)
    ablation_results.append({
        'Model': 'Conventional Data + SVM',
        'RMSE (tons/ha)': np.sqrt(mean_squared_error(y_yield_test, y_pred_yield_svm_conv)),
        'Accuracy': accuracy_score(y_stress_test, y_pred_stress_svm_conv)
    })

    # --- Federated Learning Simulation (Full QAL Model) ---
    print("Running Federated Simulation for Full QAL Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    client_data_partitions = np.array_split(train_featured, config['n_clients'])
    
    num_features_full = X_train_scaled.shape[1]
    global_model = FSPM_PyTorch(num_features_full, config).to(device)
    convergence_data = []
    
    for round_num in tqdm(range(config['federated_rounds']), desc="Federated Rounds"):
        client_weights = []
        for i in range(config['n_clients']):
            local_model = deepcopy(global_model)
            optimizer = torch.optim.AdamW(local_model.parameters(), lr=config['fspm_hyperparams']['lr'])
            
            client_df = client_data_partitions[i]
            X_client_scaled = scaler.transform(client_df[feature_cols])
            
            client_dataset = TensorDataset(
                torch.tensor(X_client_scaled, dtype=torch.float32),
                torch.tensor(client_df['yield'].values, dtype=torch.float32),
                torch.tensor(client_df['stress_class'].values, dtype=torch.long)
            )
            client_loader = DataLoader(client_dataset, batch_size=config['fspm_hyperparams']['batch_size'], shuffle=True)
            
            weights = _client_update(local_model, optimizer, client_loader, config['local_epochs'], device, config)
            client_weights.append(weights)
        
        global_model = _server_aggregate(global_model, client_weights)
        
        test_dataset = TensorDataset(
            torch.tensor(X_test_scaled, dtype=torch.float32),
            torch.tensor(y_yield_test, dtype=torch.float32),
            torch.tensor(y_stress_test, dtype=torch.long)
        )
        test_loader = DataLoader(test_dataset, batch_size=config['fspm_hyperparams']['batch_size'])
        
        rmse, acc, _, _, _, _, _ = _evaluate_fspm_model(global_model, test_loader, device)
        convergence_data.append({'round': round_num + 1, 'accuracy': acc})

    # Final evaluation of the Full QAL model
    final_rmse, final_acc, y_true_yield, y_pred_yield_qal, y_true_stress, y_pred_stress_qal, y_pred_stress_probs_qal = _evaluate_fspm_model(global_model, test_loader, device)
    
    ablation_results.append({
        'Model': 'Full Data + FSPM (QAL)',
        'RMSE (tons/ha)': final_rmse,
        'Accuracy': final_acc
    })

    return {
        "ablation_results": ablation_results,
        "convergence": convergence_data,
        "predictions": {
            'y_true_yield': y_true_yield,
            'y_pred_yield_qal': y_pred_yield_qal,
            'y_pred_yield_baseline': y_pred_yield_rf_conv,
            'y_true_stress': y_true_stress,
            'y_pred_stress_qal': y_pred_stress_qal,
            'y_pred_stress_probs_qal': y_pred_stress_probs_qal
        },
        'fspm_full': {'rmse': final_rmse, 'accuracy': final_acc}
    }