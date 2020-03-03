import torch
from torch import optim
import torch.nn as nn
from collections import OrderedDict
import copy
from torch.distributions.normal import Normal
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'

class PiNetwork(nn.Module):
    def __init__(self, input_size, output_size, layer_size1=64, layer_size2=64, lr=1e-5, seed=1337):
        super(PiNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_size = input_size
        self.output_size = output_size
        self.layer_size1 = layer_size1
        self.layer_size2 = layer_size2
        self.lr = lr

        self.model = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, self.layer_size1)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.layer_size1, self.layer_size2)),
            ('relu2', nn.ReLU()),
            ('logits', nn.Linear(self.layer_size2, self.output_size)),
            ('tanh', nn.Tanh())
        ]))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, state):
        return self.model.forward(state).reshape(-1, self.output_size)

    def clone(self):
        pi_network_clone = PiNetwork(self.input_size, self.output_size, self.layer_size1, self.layer_size2, self.lr)
        pi_network_clone.model = copy.deepcopy(self.model)
        return pi_network_clone


class PiNetworkStochastic(nn.Module):
    def __init__(self, input_size, output_size, layer_size1=64, layer_size2=64, lr=1e-5, seed=1337):
        super(PiNetworkStochastic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_size = input_size
        self.output_size = output_size
        self.layer_size1 = layer_size1
        self.layer_size2 = layer_size2
        self.lr = lr

        self.model = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, self.layer_size1)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.layer_size1, self.layer_size2)),
            ('relu2', nn.ReLU()),
            ('logits', nn.Linear(self.layer_size2, self.output_size)),
            ('tanh', nn.Tanh())
        ]))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, state):
        mu_sigma = self.forward_pass(state)
        m = Normal(mu_sigma[:, :4], 0.2*(0.5 + 0.5 * mu_sigma[:, 4:]))
        actions = m.sample()
        return actions

    def forward_pass(self, state):
        return self.model.forward(state).reshape(-1, self.output_size)

    def clone(self):
        pi_network_clone = PiNetworkStochastic(self.input_size, self.output_size,
                                               self.layer_size1, self.layer_size2, self.lr)
        pi_network_clone.model = copy.deepcopy(self.model)
        return pi_network_clone

    def act(self, state):
        mu_sigma = self.forward_pass(state)

        m = Normal(mu_sigma[:, :4], 0.2*(0.5 + 0.5 * mu_sigma[:, 4:]))
        actions = m.sample()
        log_probs = m.log_prob(actions)
        actions = actions
        return np.clip(actions, -1, 1), log_probs.sum(dim=1).reshape(-1, 1)
