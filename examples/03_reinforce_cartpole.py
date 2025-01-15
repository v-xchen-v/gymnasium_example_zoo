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

import gymnasium as gym

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

episode_over = False
while not episode_over:
    action = env.action_space.sample() # agent policy that uses the observation and info
    next_observation, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated
env.close()