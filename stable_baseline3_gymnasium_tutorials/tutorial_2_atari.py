import gymnasium as gym
import os
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import ale_py

Test_env = False
Train_agent = False
Evaluate_agent = True
Use_the_agent = True

# --- Loading an environment --- #
environment_name = "ALE/Breakout-v5" 
print(f"[INFO] Loading {environment_name} environment")
env = gym.make(environment_name)
test_env = gym.make(environment_name, render_mode="human") # Create an environment with the render mode set to human (render the environment)
test_env.metadata['render_fps'] = 30 # set fps for rendering

# Analyize the enviroment
print(f"[INFO] Obs Space: {env.observation_space}")
print(f"[INFO] Act Space: {env.action_space}")


# --- Test the environment --- #
if Test_env:
    print('[INFO] Testing the environment..')
    episodes = 5
    for episode in range(1, episodes+1):
        state = test_env.reset()
        done = False
        score = 0

        while not done:
            test_env.render()
            action = env.action_space.sample()
            obs, reward, done, trun, info = test_env.step(action)
            score += reward
            
        print(f'Episode: {episode}, Score: {score}')
    test_env.close()

# -- Training an Agent -- #
# First, create folders to save the logs and the models (folder Training, subfolders logs and Save Models)
log_path = os.path.join("Training", "logs")
PPO_path = os.path.join("Training", "Save_Models", "PPO_Model_Breakout")

envs = DummyVecEnv([lambda: env]) # The environment must be wrapped in DummyVecEnv (vectorized environment) to be used with Stable Baselines. Lambda is used to create a function that returns the environment.
                                 # Here we did not vectorize the environment, see Tutorial 2. vectorize = run multiple environments in parallel, speeding up training
if Train_agent:
    print("[INFO] Training the agent..")
    model = PPO("CnnPolicy", envs, verbose=1, tensorboard_log=log_path) # Create a PPO model: agent with a Multi-Layer Perceptron (Mlp) policy (structure of the NN) and the environment. (verbose = 1) to see the training process.
                                                                         # The tensorboard_log parameter allows to log the training process and visualize it in tensorboard.

    # help(PPO) # To see the documentation of the model
    
    model.learn(total_timesteps=20000) # Train the model for 20000 timesteps (1 timestep = 1 step in the environment)
    model.save(PPO_path) # Save the model in the specified
else:
    print("[INFO] Skipping training..")
    model = PPO.load(PPO_path, env=envs) # Load the model from the specified path
    
# # --- Test the trained agent --- #
# if Evaluate_agent:
#     print("[INFO] Evaluating the agent..")
#     mean_reward, std_dev = evaluate_policy(model, env, n_eval_episodes=10, render=False) # Evaluate the model on the environment for 10 episodes and render the environment
#     print(f'Mean reward: {mean_reward}, Standard deviation: {std_dev}')
#      # output: mean reward and standard deviation. It gets +1 for each step the pole is up, and zero otherwise. The maximum score is 500 (max 500 steps per episode).
     
     
# --- Use the trained agent --- #
if Use_the_agent:
    print("[INFO] Using the agent..")
    episodes = 5
    for episode in range(1, episodes+1):
        obs, _ = test_env.reset()
        done = False
        score = 0

        while not done:
            test_env.render()
            action, _ = model.predict(obs)
            obs, reward, done, trun, info = test_env.step(action)
            score += reward
        
        print(f'Episode: {episode}, Score: {score}')
    test_env.close()