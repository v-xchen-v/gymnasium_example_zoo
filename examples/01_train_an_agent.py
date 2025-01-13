# Ref: https://gymnasium.farama.org/introduction/train_agent/
"""QLearning to solve the blackjack environment."""


import gymnasium as gym

n_episode = 100
env = gym.make('Blackjack-v1', sab=False, render_mode='human')


from tqdm import tqdm
for episode in tqdm(range(n_episode)):
    obs, info = env.reset() 
    done = False
    
    # play one episode
    while not done:
        # action = agent.get_action(obs)
        action = env.action_space.sample()
        next_obs, reward, terminated, trucated, info = env.step(action)
        
        # update the agent
        # agent.update(obs, action, reward, next_obs, done)
        
        # update if the environment is done and the current obs
        done = terminated or trucated
        obs = next_obs
    
    # agent.decay_epsilon()