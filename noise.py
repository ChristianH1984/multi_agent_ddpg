import random
import copy
import numpy as np


class OUNoise:
	"""Ornstein-Uhlenbeck process."""

	def __init__(self, action_size, num_agents, seed, mu=0.0, theta=0.05, sigma=0.2, sigma_decay=0.995, sigma_min=0.0001):
		"""Initialize parameters and noise process."""
		self.mu = mu * np.ones((num_agents, action_size))
		self.theta = theta
		self.sigma = sigma
		self.seed = random.seed(seed)
		self.state = copy.copy(self.mu)
		self.sigma_decay = sigma_decay
		self.sigma_min = sigma_min

	def reset(self):
		"""Reset the internal state (= noise) to mean (mu)."""
		self.state = copy.copy(self.mu)
		if self.sigma > self.sigma:
			self.sigma *= self.sigma_min

	def sample(self):
		"""Update internal state and return it as a noise sample."""
		x = self.state
		dx = self.theta * (self.mu - x) + self.sigma * (np.random.rand(*x.shape)-0.5)
		self.state = x + dx
		return self.state

class GaussianNoise:

	def __init__(self, action_size, num_agents, seed, mu=0.0, sigma=0.2, sigma_decay=0.995, sigma_min=0.0001):
		self.mu = mu
		self.sigma = sigma
		self.seed = random.seed(seed)
		self.sigma_decay = sigma_decay
		self.sigma_min = sigma_min
		self.action_size = action_size
		self.num_agents = num_agents

	def reset(self):
		if self.sigma > self.sigma:
			self.sigma *= self.sigma_min

	def sample(self):
		return self.mu + np.random.normal(self.mu, self.sigma, (self.num_agents, self.action_size))

