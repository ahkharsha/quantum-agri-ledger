import torch
import torch.nn as nn

class FSPM_PyTorch(nn.Module):
    """
    PyTorch implementation of the Federated Stress-Phenotyping Model (FSPM).
    """
    def __init__(self, num_features, config):
        super(FSPM_PyTorch, self).__init__()
        hyperparams = config['fspm_hyperparams']
        
        self.network = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, hyperparams['layer1_neurons']),
            nn.ReLU(),
            nn.Dropout(hyperparams['dropout_rate']),
            nn.BatchNorm1d(hyperparams['layer1_neurons']),
            nn.Linear(hyperparams['layer1_neurons'], hyperparams['layer2_neurons']),
            nn.ReLU(),
            nn.Dropout(hyperparams['dropout_rate'])
        )

        # Output head for yield prediction (regression)
        self.yield_head = nn.Linear(hyperparams['layer2_neurons'], 1)

        # Output head for stress classification
        self.stress_head = nn.Linear(hyperparams['layer2_neurons'], len(config['stress_classes']))

    def forward(self, x):
        features = self.network(x)
        yield_output = self.yield_head(features)
        stress_output = self.stress_head(features)
        return yield_output, stress_output