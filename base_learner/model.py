import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fcn_sizes=[64,64,64,64], seed=42):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.observation_ct = 0
        layer_sizes = [state_size] + fcn_sizes + [action_size]
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(torch.nn.ReLU())
        layers = layers[:-1] # remove the final relu layer
        self.model = torch.nn.Sequential(*layers)


    def forward(self, state):
        """Build a network that maps state -> action values."""

        return self.model(state)
