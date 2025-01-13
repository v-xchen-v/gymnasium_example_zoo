# Ref: https://gymnasium.farama.org/introduction/train_agent/
"""QLearning to solve the blackjack environment."""


import gymnasium as gym
import numpy as np
from collections import defaultdict

n_episode = 100_000
Render_ON = False
if Render_ON:
    env = gym.make('Blackjack-v1', sab=False, render_mode='human')
else:
    env = gym.make('Blackjack-v1', sab=False)
# record the episode statistics
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episode)

# create the agent
class BlackjackAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        self.epsilon = 1.0
        self.discount_factor = 0.95
        self.lr = 0.01
        self.training_error = []
    
    def get_action(self, obs):
        # with probability epsilon return a random action to explore the environment
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        # with the probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_table[obs]))
    
    def update(self, 
               obs,
               action, 
               reward, 
               next_obs,
               terminated):
        """Update the Q-table value of an action"""
        future_q_value = (not terminated) * np.max(self.q_table[next_obs])
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_table[obs][action]
        
        self.q_table[obs][action] += self.lr * temporal_difference
        self.training_error.append(temporal_difference)
    
    def decay_epsilon(self):
        self.epsilon = max(0.1, self.epsilon - 1e-5)

# hyperparameters
agent = BlackjackAgent(env)

# loop over episodes
from tqdm import tqdm
for episode in tqdm(range(n_episode)):
    obs, info = env.reset() 
    done = False
    
    # play one episode
    while not done:
        action = agent.get_action(obs)
        # action = env.action_space.sample()
        next_obs, reward, terminated, trucated, info = env.step(action)
        
        # update the agent
        agent.update(obs, action, reward, next_obs, terminated)
        
        # update if the environment is done and the current obs
        done = terminated or trucated
        obs = next_obs
    
    agent.decay_epsilon()
    
# visualize the training reward
from matplotlib import pyplot as plt
fig, axs = plt.subplots(1, 3, figsize=(20, 8))

axs[0].plot(np.convolve(env.return_queue, np.ones(100)))
axs[0].set_title('Episode Rewards')
axs[0].set_xlabel('Episode')
axs[0].set_ylabel('Reward')

axs[1].plot(np.convolve(env.length_queue, np.ones(100)))
axs[1].set_title("Episode Lengths")
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Length")

axs[2].plot(np.convolve(agent.training_error, np.ones(100)))
axs[2].set_title("Training Error")
axs[2].set_xlabel("Episode")
axs[2].set_ylabel("Temporal Difference")

plt.tight_layout()
plt.show()