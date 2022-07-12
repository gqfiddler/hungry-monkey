import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, normalize_ft=False):
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
        self.normalize_ft = normalize_ft
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_size, state_size*2),
            torch.nn.ReLU(),
            torch.nn.Linear(state_size*2, action_size)
            # torch.nn.Flatten(0, 1)
        )



    def forward(self, state):
        """Build a network that maps state -> action values."""

        return self.model(state)
