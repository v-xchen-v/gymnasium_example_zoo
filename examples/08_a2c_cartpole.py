import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        # Actor (Policy Network)
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic (Value Function)
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        x = self.shared_layers(state)
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value

class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=0.002, gamma=0.99):
        self.gamma = gamma
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()  # For critic loss
        self.training_error = {
            'actor_loss': [],
            'critic_loss': []
        }

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs, _ = self.model(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)

    def update(self, trajectory):
        states, actions, log_probs, rewards, dones, next_states = zip(*trajectory)
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        log_probs = torch.stack(log_probs)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        _, state_values = self.model(states)
        _, next_state_values = self.model(next_states)

        # Compute TD Target and Advantage
        td_target = rewards + self.gamma * next_state_values.squeeze() * (1 - dones)
        advantage = td_target - state_values.squeeze()

        # Policy Loss (Actor Loss)
        actor_loss = -torch.mean(log_probs * advantage.detach())
        self.training_error['actor_loss'].append(actor_loss.item())

        # Critic Loss
        critic_loss = self.loss_fn(state_values.squeeze(), td_target.detach())
        self.training_error['critic_loss'].append(critic_loss.item())

        # Total Loss and Update
        loss = actor_loss + 0.5 * critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
n_episode = 500
batch_size = 5  # Number of steps before updating

# Training the A2C agent
env = gym.make("CartPole-v1")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episode)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = A2CAgent(state_dim, action_dim)


for episode in range(n_episode):
    state = env.reset()
    state = state[0] if isinstance(state, tuple) else state  # Handle different Gym versions
    episode_reward = 0
    trajectory = []

    for t in range(200):  # Max steps per episode
        action, log_prob = agent.choose_action(state)
        next_state, reward, done, _, _ = env.step(action)

        trajectory.append((state, action, log_prob, reward, done, next_state))
        state = next_state
        episode_reward += reward

        if done or len(trajectory) >= batch_size:
            agent.update(trajectory)
            trajectory = []  # Reset batch

        if done:
            break

    print(f"Episode {episode + 1}, Reward: {episode_reward}")

# visualize the training reward
from matplotlib import pyplot as plt
fig, axs = plt.subplots(1, 4, figsize=(20, 8))

axs[0].plot(np.convolve(env.return_queue, np.ones(100)/100, mode='valid'))
axs[0].set_title('Episode Rewards')
axs[0].set_xlabel('Episode')
axs[0].set_ylabel('Reward')

axs[1].plot(np.convolve(env.length_queue, np.ones(100)/100, mode='valid'))
axs[1].set_title("Episode Lengths")
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Length")

axs[2].plot(np.convolve(agent.training_error['actor_loss'], np.ones(100)/100, mode='valid'))
axs[2].set_title("Training Error")
axs[2].set_xlabel("Episode")
axs[2].set_ylabel("Actor Loss")

axs[3].plot(np.convolve(agent.training_error['critic_loss'], np.ones(100)/100, mode='valid'))
axs[3].set_title("Training Error")
axs[3].set_xlabel("Episode")
axs[3].set_ylabel("Critic Loss")

plt.tight_layout()
plt.show()

env.close()
