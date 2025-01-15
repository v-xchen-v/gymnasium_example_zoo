# Ref: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py
"""QLearning to solve the blackjack environment."""


import gymnasium as gym
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.buffers import ReplayBuffer

n_episode = 100_000
Render_ON = False
if Render_ON:
    env = gym.make('Blackjack-v1', sab=False, render_mode='human')
else:
    env = gym.make('Blackjack-v1', sab=False)
# record the episode statistics
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episode)

class ValueNetwork(nn.Module):
    """QNetwork, given an observation, output the Q value of each action."""
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(env.observation_space), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n)
        )
    def forward(self, x):
        return self.network(x)

# create the agent
class BlackjackAgent:
    def __init__(self, env, device, buffer_size=512, batch_size=32):
        self.env = env
        self.epsilon = 1.0
        self.discount_factor = 0.95
        self.lr = 0.01
        self.training_error = []
        self.device = device
        self.q_network = ValueNetwork(env).to(device)
        self.target_network = ValueNetwork(env).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.batch_size = batch_size
        
        self.rb = ReplayBuffer(
            buffer_size=buffer_size,
            observation_space=gym.spaces.MultiDiscrete([21, 10, 1]), # Example: x in [0,21], y in [0,10] z in [0,1]
            action_space=gym.spaces.Discrete(env.action_space.n), # gym.spaces.Discrete(2), Example action space with two possible actions
            device=device,
            handle_timeout_termination=False,
        )
    
    
    def get_action(self, obs):
        # with probability epsilon return a random action to explore the environment
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        # with the probability (1 - epsilon) act greedily (exploit)
        else:
            q_values = self.q_network(torch.Tensor(obs).to(self.device))
            actions = torch.argmax(q_values, dim=0).cpu().numpy()
            return actions        
    
    def update(self, 
               obs,
               action, 
               reward, 
               next_obs,
               terminated,
               info):
        """Update the Q network with the new observation."""
        self.rb.add(np.array(obs), np.array(next_obs), action, reward, terminated, info)
        if self.rb.size() > self.batch_size:
            data = self.rb.sample(self.batch_size)
            with torch.no_grad():
                target_max, _ = self.target_network(data.next_observations.to(torch.float)).max(dim=1)
                td_target = data.rewards.flatten().to(torch.float) + self.discount_factor * target_max * (1 - data.dones.flatten().to(torch.float))
            old_values = self.q_network(data.observations.to(torch.float)).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_values)
            self.training_error.append(loss.item())
            
            # optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            tau=1.0
            for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_network_param.data.copy_(
                    tau * q_network_param.data + (1.0 - tau) * target_network_param.data
                )
    
    def decay_epsilon(self):
        self.epsilon = max(0.1, self.epsilon - 1e-5)

# hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = BlackjackAgent(env,
                       device=device,)

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
        agent.update(obs, action, reward, next_obs, terminated, info)
        
        # update if the environment is done and the current obs
        done = terminated or trucated
        obs = next_obs
    
    agent.decay_epsilon()
    
# visualize the training reward
from matplotlib import pyplot as plt
fig, axs = plt.subplots(1, 3, figsize=(20, 8))

axs[0].plot(np.convolve(env.return_queue, np.ones(100)/100))
axs[0].set_title('Episode Rewards')
axs[0].set_xlabel('Episode')
axs[0].set_ylabel('Reward')

axs[1].plot(np.convolve(env.length_queue, np.ones(100)/100))
axs[1].set_title("Episode Lengths")
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Length")

axs[2].plot(np.convolve(agent.training_error, np.ones(100)/100))
axs[2].set_title("Training Error")
axs[2].set_xlabel("Episode")
axs[2].set_ylabel("Temporal Difference")

plt.tight_layout()
plt.show()