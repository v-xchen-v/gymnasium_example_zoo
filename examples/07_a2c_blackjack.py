import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the Policy (Actor) and Value (Critic) network
class ActorCritic(nn.Module):
    # def __init__(self, state_dim, action_dim):
    #     super(ActorCritic, self).__init__()
    #     # self.fc = nn.Linear(state_dim, 128)
    #     self.fc = nn.Sequential(
    #         nn.Linear(state_dim, 128),
    #         nn.ReLU(),
    #         nn.Linear(128, 128),
    #     )
        
    #     # Actor network
    #     self.actor = nn.Linear(128, action_dim)
        
    #     # Critic network
    #     self.critic = nn.Linear(128, 1)
        
    # def forward(self, x):
    #     x = torch.relu(self.fc(x))
    #     action_probs = torch.softmax(self.actor(x), dim=-1)
    #     state_value = self.critic(x)
    #     return action_probs, state_value

    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor_fc = nn.Linear(state_dim, 128)
        self.critic_fc = nn.Linear(state_dim, 128)
        
        # Actor network
        self.actor = nn.Linear(128, action_dim)
        
        # Critic network
        self.critic = nn.Linear(128, 1)
        
    def forward(self, x):
        actor_x = torch.relu(self.actor_fc(x))
        critic_x = torch.relu(self.critic_fc(x))
        
        action_probs = torch.softmax(self.actor(actor_x), dim=-1)
        state_value = self.critic(critic_x)
        
        return action_probs, state_value

# Function to preprocess the Blackjack state

def preprocess_state(state):
    
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

def select_action(policy, state):
    action_probs, _ = policy(state)
    action = torch.multinomial(action_probs, 1).item()
    return action, action_probs[0, action]

# Training parameters
gamma = 0.99  # Discount factor
lr = 3e-4  # Learning rate
n_episode = 10000 # Number of episodes

env = gym.make("Blackjack-v1", natural=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episode)
training_error = {
    'actor_loss': [],
    'critic_loss': []
}

state_dim = len(env.observation_space.shape) if env.observation_space.shape else 3
action_dim = env.action_space.n

policy = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=lr)

for episode in range(n_episode):
    state, _ = env.reset()
    state = preprocess_state(state)
    log_probs = []
    rewards = []
    values = []
    done = False
    
    while not done:
        action, log_prob = select_action(policy, state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = preprocess_state(next_state)
        
        _, state_value = policy(state)
        
        log_probs.append(torch.log(log_prob))
        rewards.append(reward)
        values.append(state_value)
        
        state = next_state
    
    # Compute advantages and returns
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    values = torch.cat(values).squeeze()
    
    advantages = returns - values.detach()
    
    # actor_loss encourages the policy to increase the probability of actions
    # that lead to higher advantages
    actor_loss = -torch.sum(torch.stack(log_probs) * advantages)
    # critic_loss is calculated using the MSE loss between values and return,
    # which helps the critic network to better estimate the value function.
    critic_loss = torch.nn.functional.mse_loss(values, returns)
    loss = actor_loss + critic_loss
    training_error['actor_loss'].append(actor_loss.item())
    training_error['critic_loss'].append(critic_loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if episode % 500 == 0:
        print(f"Episode {episode}, Loss: {loss.item():.4f}")
        
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

axs[2].plot(np.convolve(training_error['actor_loss'], np.ones(100)/100, mode='valid'))
axs[2].set_title("Training Error")
axs[2].set_xlabel("Episode")
axs[2].set_ylabel("Actor Loss")

axs[3].plot(np.convolve(training_error['critic_loss'], np.ones(100)/100, mode='valid'))
axs[3].set_title("Training Error")
axs[3].set_xlabel("Episode")
axs[3].set_ylabel("Critic Loss")

plt.tight_layout()
plt.show()
env.close()
