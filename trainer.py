import numpy as np


class Trainer(object):
	def __init__(self, env, agent):
		self.env = env
		self.agent = agent

	def train(self, max_iter):
		brain_name = self.env.brain_names[0]
		scores = []

		for i in range(max_iter):
			score = []
			env_info = self.env.reset(train_mode=True)[brain_name]
			state = env_info.vector_observations
			self.agent.reset()
			done = [[False]]
			while not np.any(done):
				action = self.agent.act(state)
				env_info = self.env.step(action.reshape(-1))[brain_name]
				next_state, reward, done = env_info.vector_observations, env_info.rewards, env_info.local_done
				reward = np.array(reward).reshape(next_state.shape[0], -1)
				done = np.array(done).reshape(next_state.shape[0], -1)
				self.agent.step(state, action, reward, next_state, done)
				state = next_state
				score.append(reward)

			scores.append(np.max((np.sum(np.array(score).reshape(-1, reward.shape[0]), axis=0))))

			if np.mean(scores[-100:]) > 1.0:
				print("\nProblem solved after {0} episodes".format(i))
				return scores
			print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores[-20:])), end="")
			if i % 100 == 0:
				print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores[-20:])))

		return scores
