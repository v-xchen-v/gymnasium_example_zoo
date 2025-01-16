import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import panda_gym

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            # nn.Softmax(dim=-1)
        )
        
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    def bn(self, x):
        # normalize the input
        return (x - x.mean()) / (x.std() + 1e-8)
        
    def forward(self, x):
        # Normalize inputs
        x = self.bn(x)
        
        logits = self.fc(x)
        logits = logits - logits.max(dim=-1, keepdim=True).values  # Numerical stability
        if torch.any(torch.isnan(logits)):
            raise ValueError("NaN in probabilities")
        return torch.softmax(logits, dim=-1)

# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, eps_clip=0.2):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

    def select_action(self, state):
        # [1, 6] > [6]
        state = torch.tensor(state, dtype=torch.float32).to(self.device)  # Batch of 1
        probs = self.policy(state)
        if torch.any(torch.isnan(probs)):
            raise ValueError("NaN encountered in action probabilities")
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns

    def update(self, memory):
        states = torch.tensor(memory['states'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(memory['actions']).to(self.device)
        log_probs = torch.tensor(memory['log_probs']).to(self.device)
        rewards = memory['rewards']

        returns = self.compute_returns(rewards)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize returns

        for _ in range(10):  # PPO update iterations
            probs = self.policy(states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)

            ratios = torch.exp(new_log_probs - log_probs)
            advantages = returns - returns.mean()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)  # Gradient clipping
            self.optimizer.step()


# Train the PPO agent
if __name__ == "__main__":
    # env_id = "PandaReachDense-v3"
    env_id = "Blackjack-v1"
    # env = gym.make("Blackjack-v1", natural=True)
    env = gym.make(env_id)
    state_dim = len(env.observation_space)  # The state is represented by 3 integers
    action_dim = env.action_space.n # 2 actions: hit or stick

    agent = PPOAgent(state_dim, action_dim)
    num_episodes = 5000
    max_steps = 10

    for episode in range(num_episodes):
        state, _ = env.reset()
        memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': []}
        total_reward = 0

        for _ in range(max_steps):
            action, log_prob = agent.select_action(state)
            memory['states'].append(state)
            memory['actions'].append(action)
            memory['log_probs'].append(log_prob.item())

            state, reward, done, _, _ = env.step(action)
            memory['rewards'].append(reward)
            total_reward += reward

            if done:
                break

        agent.update(memory)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    env.close()
