import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomRobotEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Observation space (position: continuous, gripper: binary)
        self.observation_space = spaces.Tuple([
            spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            spaces.Discrete(2)
        ])
        
        # Action space (move: discrete, velocity: continuous)
        self.action_space = spaces.Tuple([
            spaces.Discrete(4), # Move: 0 (Up), 1 (Down), 2 (Left), 3 (Right)
            spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        ])
        
        
    def reset(self, seed=None):
        self.state = {"position": np.random.uniform(-1, 1, size=(2,)), "gripper_open": 0}
        return self.state, {}
    
    def step(self, action):
        move_type, velocity = action
        self.state["position"] += velocity # Apply movement
        self.state["gripper_open"] = move_type==2 # Open gripper if action is 2

        reward = -np.linalg.norm(self.state["position"])  # Example reward
        terminated = False
        return self.state, reward, terminated, False, {}
    
env = CustomRobotEnv()
obs, info = env.reset()
print(env.action_space.sample()) # Example action