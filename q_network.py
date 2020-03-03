import torch
from torch import optim
import torch.nn as nn
from collections import OrderedDict
import copy


class QNetwork(nn.Module):
	"""Actor (Policy) Model."""

	def __init__(self, input_size, layer_size1=128, layer_size2=64, lr=1e-5, seed=1337):
		"""Initialize parameters and build model.
		Params
		======
			state_size (int): Dimension of each state
			action_size (int): Dimension of each action
			seed (int): Random seed
		"""
		super(QNetwork, self).__init__()
		self.seed = torch.manual_seed(seed)
		"*** YOUR CODE HERE ***"
		self.input_size = input_size

		hidden_sizes = [layer_size1, layer_size2]

		# Build a feed-forward network
		self.model = nn.Sequential(OrderedDict([
			('fc1', nn.Linear(input_size, hidden_sizes[0])),
			('relu1', nn.ReLU()),
			('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
			('relu2', nn.ReLU()),
			('logits', nn.Linear(hidden_sizes[1], 1))
		]))

		self.loss = nn.MSELoss()
		self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

	def forward(self, state):
		"""Build a network that maps state -> action values."""
		return self.model.forward(state)

	def clone(self):
		qnetwork_clone = QNetwork(self.input_size)
		qnetwork_clone.model = copy.deepcopy(self.model)
		return qnetwork_clone
