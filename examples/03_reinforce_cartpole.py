#Ref:https://huggingface.co/learn/deep-rl-course/en/unit4/hands-on

'''
CartPole-v1 environment(https://gymnasium.farama.org/environments/classic_control/cart_pole/):
Action Space:
Type: Discrete(2)
0: Push cart to the left
1: Push cart to the right

Observation Space:
Type: Box(4)
Num Observation Min Max
0 Cart Position -4.8 4.8
1 Cart Velocity -Inf Inf
2 Pole Angle -24 deg 24 deg
3 Pole Angular Velocity -Inf Inf

Reward:
Since the goal is to keep the pole upright for as long as possible, by default, a reward of +1 is given for every step taken, 
including the termination step. The default reward threshold is 500 for v1 and 200 for v0 due to the time limit on the environment.

The episode ends if:
1. Termination: Pole Angle is greater than ±12°
2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
3. Truncation: Episode length is greater than 500 (200 for v0)
'''
from collections import deque

import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

env_id = "CartPole-v1"
env = gym.make(env_id, render_mode='human')
observation, info = env.reset()

# get the state space and action space
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", state_size)
print("Sample observation", env.observation_space.sample())  # Get a random observation
print("\n _____ACTION SPACE_____ \n")
print("The Action Space is: ", action_size)
print("Action Space Sample", env.action_space.sample())  # Take a random action

class Policy(nn.Module):
    def __init__(self, device, state_size, action_size, h_size=24):
        super(Policy, self).__init__()
        self.device = device
        
        # create two fully connected layers
        self.fc1 = nn.Linear(state_size, h_size)
        self.fc2 = nn.Linear(h_size, action_size)
    def forward(self, x):
        # define the forward pass
        # state goes through fc1, then through ReLU activation
        x = F.relu(self.fc1(x))
        # fc1 output goes through fc2
        x = self.fc2(x)
        # we output the softmax
        return F.softmax(x, dim=1)
    
    def get_action(self, state):
        # convert the state to a tensor
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # get the action probabilities
        probs = self.forward(state).cpu()
        # sample an action
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
def reinforce(policy: Policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # Line 3 of pseudocode
    for i_episode in range(1, n_training_episodes+1):
        saved_log_probs = []
        rewards = []
        state, info = env.reset() # reset to the initial state
        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.get_action(state)
            saved_log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            
            
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Line 6 of pseudocode: calculate the return
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)
        # Compute the discounted returns at each timestep,
        # as the sum of the gamma-discounted return at time t (G_t) + the reward at time t

        # In O(N) time, where N is the number of time steps
        # (this definition of the discounted return G_t follows the definition of this quantity
        # shown at page 44 of Sutton&Barto 2017 2nd draft)
        # G_t = r_(t+1) + r_(t+2) + ...

        # Given this formulation, the returns at each timestep t can be computed
        # by re-using the computed future returns G_(t+1) to compute the current return G_t
        # G_t = r_(t+1) + gamma*G_(t+1)
        # G_(t-1) = r_t + gamma* G_t
        # (this follows a dynamic programming approach, with which we memorize solutions in order
        # to avoid computing them multiple times)

        # This is correct since the above is equivalent to (see also page 46 of Sutton&Barto 2017 2nd draft)
        # G_(t-1) = r_t + gamma*r_(t+1) + gamma*gamma*r_(t+2) + ...


        ## Given the above, we calculate the returns at timestep t as:
        #               gamma[t] * return[t] + reward[t]
        #
        ## We compute this starting from the last timestep to the first, in order
        ## to employ the formula presented above and avoid redundant computations that would be needed
        ## if we were to do it from first to last.

        ## Hence, the queue "returns" will hold the returns in chronological order, from t=0 to t=n_steps
        ## thanks to the appendleft() function which allows to append to the position 0 in constant time O(1)
        ## a normal python list would instead require O(N) to do this.
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft(gamma*disc_return_t+rewards[t])
        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()

        ## eps is the smallest representable float, which is
        # added to the standard deviation of the returns to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Line 7:
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        # Line 8: PyTorch prefers gradient descent
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    return scores
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cartpole_hyparameters = {
    "h_size": 16,
    "n_training_episodes": 1000,
    "n_evaluation_episodes": 100,
    "max_t": 1000,
    "gamma": 1.0,
    "lr": 1e-2,
    "env_id": env_id,
    "state_space": state_size,
    "action_space": action_size,
}
policy = Policy(device,
                state_size=cartpole_hyparameters["state_space"],
                action_size=cartpole_hyparameters["action_space"],
                h_size=cartpole_hyparameters["h_size"]).to(device)
cartpole_optimizer = torch.optim.Adam(policy.parameters(), lr=cartpole_hyparameters["lr"])
scores = reinforce(policy, 
            cartpole_optimizer, 
            cartpole_hyparameters["n_training_episodes"], 
            cartpole_hyparameters["max_t"],
            cartpole_hyparameters["gamma"], 
            print_every=100)

def evaluate_policy(env, policy, n_episodes, max_t):
    scores = []
    for i_episode in range(1, n_episodes+1):
        state, info = env.reset()
        total_rewards_ep = 0
        for t in range(max_t):
            action, _ = policy.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_rewards_ep += reward
            if terminated or truncated:
                break
        scores.append(total_rewards_ep)
    main_rewards = np.mean(scores)
    std_rewards = np.std(scores)
    return main_rewards, std_rewards

main_rewards, std_rewards = evaluate_policy(env, policy, 
                cartpole_hyparameters["n_evaluation_episodes"], 
                cartpole_hyparameters["max_t"])
print("Main rewards: ", main_rewards)
print("Standard deviation of the rewards: ", std_rewards)