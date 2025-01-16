import gymnasium as gym
import panda_gym

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3.common.vec_env import VecNormalize

env_id = "PandaReachDense-v3"
# env = gym.make(env_id, render_mode="human")
env = make_vec_env(env_id, n_envs=4)

env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
# Get the state space and action space
s_size = env.observation_space.shape
a_size = env.action_space
print("______OBSERVATION SPACE______")
print(f"The State Space is: {s_size}")
print("Sample observation", env.observation_space.sample()) # Get a random observation
print(f"The Action Space is: {a_size}")
print(f"Action Space Sample, {env.action_space.sample()}") # Get a random action

# observation, info = env.reset()

model = A2C('MultiInputPolicy', env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("a2c_PandaReachDense-v3")
env.save("vec_normalize.pkl")
# for _ in range(1000):
#     action = env.action_space.sample() # random action
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()

# env.close()