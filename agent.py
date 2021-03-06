import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'


class BaseAgent:
	"""Interacts with and learns from the environment."""

	def __init__(self, gamma, tau, batch_size, update_every, actor, critic, memory, noise):
		"""Initialize an Agent object.

		Params
		======
			gamma (float): discount factor of future rewards
			tau (float): soft update parameter
			batch_size (int): number of samples for each mini batch
			update_every (int): number of steps until weights are copied  from local to target Q-network
			state_size (int): dimension of each state
			action_size (int): dimension of each action
			seed (int): random seed
		"""
		self.gamma = gamma
		self.tau = tau
		self.batch_size = batch_size
		self.update_every = update_every
		self.critic_local = critic
		self.critic_target = critic.clone()

		self.actor_local = actor
		self.actor_target = actor.clone()

		self.noise = noise

		# Replay memory
		self.memory = memory

		# Initialize time step (for updating every UPDATE_EVERY steps)
		self.t_step = 0

	def step(self, state, action, reward, next_state, done):
		# Save experience in replay memory
		self.memory.add(state, action, reward, next_state, done)

		# Learn every UPDATE_EVERY time steps.
		self.t_step = (self.t_step + 1) % self.update_every
		if self.t_step == 0:
			# If enough samples are available in memory, get random subset and learn
			if len(self.memory) > self.batch_size:
				experiences = self.memory.sample()
				self.learn(experiences)

	def act(self, state):
		"""Returns actions for given state as per current policy.

		Params
		======
			state (array_like): current state
			eps (float): epsilon, for epsilon-greedy action selection
		"""
		state = torch.from_numpy(state).float().to(device)
		self.actor_local.eval()
		with torch.no_grad():
			actions = self.actor_local(state)
		self.actor_local.train()
		actions = actions.cpu().data.numpy()
		actions += self.noise.sample().reshape(-1, actions.shape[1])
		return np.clip(actions, -1, 1)

	def learn(self, experiences):
		"""Update value parameters using given batch of experience tuples.

		Params
		======
			experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
		"""
		pass

	def soft_update(self, local_model, target_model):
		"""Soft update model parameters.
		θ_target = τ*θ_local + (1 - τ)*θ_target

		Params
		======
			local_model (PyTorch model): weights will be copied from
			target_model (PyTorch model): weights will be copied to
		"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

	def reset(self):
		self.noise.reset()


class DeterministicActorCriticAgent(BaseAgent):
	"""Interacts with and learns from the environment."""

	def learn(self, experiences):
		"""Update value parameters using given batch of experience tuples.

		Params
		======
			experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
		"""
		states, actions, rewards, next_states, dones = experiences

		states = states[0]
		actions = actions[0]
		rewards = rewards[0]
		next_states = next_states[0]
		dones = dones[0]

		next_actions = self.actor_target(next_states)

		next_states_actions = torch.cat((next_states, next_actions), 1)
		states_actions = torch.cat((states.float(), actions.float()), 1)

		y_target = 100 * rewards + self.gamma * self.critic_target(next_states_actions)*(1 - dones)
		critic_error = 0.5*((y_target.detach() - self.critic_local(states_actions)) ** 2).mean()

		self.critic_local.optimizer.zero_grad()
		critic_error.backward()
		torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
		self.critic_local.optimizer.step()

		actions = self.actor_local(states)
		states_actions = torch.cat((states.float(), actions.float()), 1)
		action_error = -self.critic_local(states_actions).mean()
		self.actor_local.optimizer.zero_grad()
		action_error.backward()
		self.actor_local.optimizer.step()

		self.soft_update(self.critic_local, self.critic_target)
		self.soft_update(self.actor_local, self.actor_target)


class StochasticActorCriticAgent(BaseAgent):
	"""Interacts with and learns from the environment."""

	def learn(self, experiences):
		"""Update value parameters using given batch of experience tuples.

		Params
		======
			experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
		"""
		states, actions, rewards, next_states, dones = experiences

		next_actions = self.actor_target(next_states)

		next_states_actions = torch.cat((next_states, next_actions), 1)
		states_actions = torch.cat((states.float(), actions.float()), 1)

		y_target = 100 * rewards + self.gamma * self.critic_target(next_states_actions)*(1 - dones)
		critic_error = 0.5*((y_target.detach() - self.critic_local(states_actions)) ** 2).mean()

		self.critic_local.optimizer.zero_grad()
		critic_error.backward()
		torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
		self.critic_local.optimizer.step()

		actions, log_probs = self.actor_local.act(states)
		states_actions = torch.cat((states.float(), actions.float()), 1)
		action_error = (-log_probs * self.critic_local(states_actions).detach()).mean()
		self.actor_local.optimizer.zero_grad()
		action_error.backward()
		self.actor_local.optimizer.step()

		self.soft_update(self.critic_local, self.critic_target)
		self.soft_update(self.actor_local, self.actor_target)


class MultiAgentDeterministicActorCriticAgent(BaseAgent):
	"""Interacts with and learns from the environment."""

	def __init__(self, gamma, tau, batch_size, update_every, actors, critics, memory, noise):
		"""Initialize an Agent object.

		Params
		======
			gamma (float): discount factor of future rewards
			tau (float): soft update parameter
			batch_size (int): number of samples for each mini batch
			update_every (int): number of steps until weights are copied  from local to target Q-network
			state_size (int): dimension of each state
			action_size (int): dimension of each action
			seed (int): random seed
		"""
		self.gamma = gamma
		self.tau = tau
		self.batch_size = batch_size
		self.update_every = update_every
		self.critics_local = critics
		self.critics_target = [critic.clone() for critic in critics]

		self.actors_local = actors
		self.actors_target = [actor.clone() for actor in actors]

		self.noise = noise

		# Replay memory
		self.memory = memory

		# Initialize time step (for updating every UPDATE_EVERY steps)
		self.t_step = 0

	def act(self, state):
		"""Returns actions for given state as per current policy.

		Params
		======
			state (array_like): current state
			eps (float): epsilon, for epsilon-greedy action selection
		"""
		state = torch.from_numpy(state).float().to(device)
		for actor in self.actors_local:
			actor.eval()
		with torch.no_grad():
			actions = self._compute_next_actions(self.actors_local, state)#self.actor_local(state)
		for actor in self.actors_local:
			actor.train()
		actions = actions.cpu().data.numpy()
		actions += self.noise.sample().reshape(2, 1, 2)
		return np.clip(actions, -1, 1).reshape(2, 2)

	def learn(self, experiences):
		"""Update value parameters using given batch of experience tuples.

		Params
		======
			experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
		"""
		states, actions, rewards, next_states, dones = experiences

		next_actions = self._compute_next_actions(self.actors_target, next_states)
		next_states_actions = torch.cat((next_states.permute(1, 0, 2).reshape(self.batch_size, -1),
										 next_actions.permute(1, 0, 2).reshape(self.batch_size, -1)), 1)

		states_actions = torch.cat((states.float().permute(1, 0, 2).reshape(self.batch_size, -1),
									actions.float().permute(1, 0, 2).reshape(self.batch_size, -1)), 1)

		for i in range(len(self.critics_local)):
			self._train_critic(self.critics_local[i], self.critics_target[i], states_actions,
							   next_states_actions, rewards[i], dones[i])

		for i in range(len(self.actors_local)):
			self._train_actor(self.actors_local[i], self.actors_target[i], self.critics_local[i], states)

	def _train_actor(self, actor_local, actor_target, critics_local, states):
		actions = self._compute_next_actions(self.actors_local, states)

		states_actions = torch.cat((states.float().permute(1, 0, 2).reshape(self.batch_size, -1),
									actions.float().permute(1, 0, 2).reshape(self.batch_size, -1)), 1)

		action_error = -critics_local(states_actions).mean()
		actor_local.optimizer.zero_grad()
		action_error.backward()
		actor_local.optimizer.step()

		self.soft_update(actor_local, actor_target)

	def _train_critic(self, critic_local, critic_target, states_actions, next_states_actions, rewards, dones):
		y_target = 100 * rewards + self.gamma * critic_target(next_states_actions) * (1 - dones)
		critic_error = 0.5 * ((y_target.detach() - critic_local(states_actions)) ** 2).mean()

		critic_local.optimizer.zero_grad()
		critic_error.backward()
		torch.nn.utils.clip_grad_norm_(critic_local.parameters(), 1)
		critic_local.optimizer.step()

		self.soft_update(critic_local, critic_target)

	def _compute_next_actions(self, actors, next_states):
		return torch.stack([actors[i](next_states[i]) for i in range(len(actors))], 0)
