import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym  # Changed to gymnasium
from torch.distributions import Categorical

# Hyperparameters
GAMMA = 0.99
LR = 3e-4
EPS_CLIP = 0.2
K_EPOCHS = 4
BATCH_SIZE = 64
UPDATE_INTERVAL = 2000  # Number of steps before updating

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_action, activation=nn.Tanh):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, n_action),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 32),
            activation(),
            nn.Linear(32, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, state_value.squeeze(), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(dev)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.policy_old = ActorCritic(state_dim, action_dim).to(dev)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.mse_loss = nn.MSELoss()
        self.training_loss = []

    def update(self, memory):
        rewards, states, actions, logprobs = memory
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=dev).unsqueeze(1)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=dev)
        actions = torch.tensor(np.array(actions), dtype=torch.int64, device=dev)
        logprobs = torch.tensor(np.array(logprobs), dtype=torch.float32, device=dev).unsqueeze(1)

        for _ in range(K_EPOCHS):
            action_logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            ratios = (torch.exp(action_logprobs - logprobs.squeeze(1))).unsqueeze(1)
            advantages = rewards - state_values.detach().unsqueeze(1)
            surr1 = -ratios * advantages
            surr2 = -torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
            # loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values.unsqueeze(1), rewards) - 0.01 * dist_entropy
            loss = torch.max(surr1, surr2).mean()
            self.training_loss.append(loss.mean().item())
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())

def train():
    env = gym.make("Blackjack-v1")
    # record the episode statistics
    n_episode = 10000
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episode)


    state_dim = 3  # Blackjack state is (player sum, dealer card, usable ace)
    action_dim = env.action_space.n
    ppo = PPO(state_dim, action_dim)
    memory = [[], [], [], []]
    total_timesteps = 0

    for episode in range(n_episode):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=dev)
        ep_reward = 0
        for t in range(100):  # Max steps per episode
            action, action_logprob = ppo.policy_old.act(state)
            next_state, reward, terminated, trucated, _ = env.step(action)
            ep_reward += reward
            memory[0].append(np.array(reward))
            memory[1].append(state.cpu().numpy())
            memory[2].append(np.array(action))
            memory[3].append(action_logprob.detach().cpu().numpy())
            state = torch.tensor(next_state, dtype=torch.float32, device=dev)
            total_timesteps += 1
            if total_timesteps % UPDATE_INTERVAL == 0:
                ppo.update(memory)
                memory = [[], [], [], []]
                
            done = terminated or trucated
            if done:
                break
        print(f"Episode {episode}, Reward: {ep_reward}")
        
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

    # axs[2].plot(np.convolve(agent.training_error, np.ones(100)/100))
    # axs[2].set_title("Training Error")
    # axs[2].set_xlabel("Episode")
    # axs[2].set_ylabel("Temporal Difference")

    plt.tight_layout()
    plt.show()
    env.close()

if __name__ == "__main__":
    train()
