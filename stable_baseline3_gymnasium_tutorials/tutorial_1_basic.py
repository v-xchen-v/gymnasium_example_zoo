import gymnasium as gym
import os
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

Test_env = True
Train_agent = True
Evaluate_agent = True
Use_the_agent = True
Train_with_callbacks = False
Train_with_mod_NN = False
Train_with_diff_alg = False

# --- Loading an environment --- #
print("[INFO] Loading CartPole-v1 environment")
environment_name = "CartPole-v1" 
env = gym.make(environment_name)
test_env = gym.make(environment_name, render_mode="human") # Create an environment with the render mode set to human (render the environment)

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
PPO_path = os.path.join("Training", "Save_Models", "PPO_Model_CartPole")

envs = DummyVecEnv([lambda: env]) # The environment must be wrapped in DummyVecEnv (vectorized environment) to be used with Stable Baselines. Lambda is used to create a function that returns the environment.
                                 # Here we did not vectorize the environment, see Tutorial 2. vectorize = run multiple environments in parallel, speeding up training
if Train_agent:
    print("[INFO] Training the agent..")
    model = PPO("MlpPolicy", envs, verbose=1, tensorboard_log=log_path) # Create a PPO model: agent with a Multi-Layer Perceptron (Mlp) policy (structure of the NN) and the environment. (verbose = 1) to see the training process.
                                                                         # The tensorboard_log parameter allows to log the training process and visualize it in tensorboard.

    # help(PPO) # To see the documentation of the model
    
    model.learn(total_timesteps=20000) # Train the model for 20000 timesteps (1 timestep = 1 step in the environment)
    model.save(PPO_path) # Save the model in the specified
else:
    print("[INFO] Skipping training..")
    model = PPO.load(PPO_path, env=envs) # Load the model from the specified path
    
# --- Test the trained agent --- #
if Evaluate_agent:
    print("[INFO] Evaluating the agent..")
    mean_reward, std_dev = evaluate_policy(model, env, n_eval_episodes=10, render=True) # Evaluate the model on the environment for 10 episodes and render the environment
    print(f'Mean reward: {mean_reward}, Standard deviation: {std_dev}')
     # output: mean reward and standard deviation. It gets +1 for each step the pole is up, and zero otherwise. The maximum score is 500 (max 500 steps per episode).
     
     
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
    

###################################################
## Visualize the training process in tensorboard ##
###################################################
# with $tensorboard --logdir Training/logs/xxx

###############
## Callbacks ##
###############
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# Callbacks are used to save the model, log the training process, and stop the training when a certain condition is met.
# Useful when you have large training processes and you want to save the model at certain intervals, log the training process, and stop the training when the model is not learning anymore.
print('Definining callbacks')
save_path = os.path.join('Training', 'Saved Models')

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1) # Stop the training when the mean reward is greater than 300

# eval callback is used to evaluate the model at certain intervals and stop the training when the mean reward is greater than 300 (stop_callback)
# Every time a new best model is found, the stop_callback is checked. If the mean reward is greater than 300, the training stops. The best model is saved in the "best_model" parameter.
# The eval callback is evaluated every "eval_freq" timesteps. The best model is saved in the "best_model" parameter.
eval_callback = EvalCallback(env, 
                            callback_on_new_best = stop_callback, 
                            eval_freq = 10000, 
                            best_model_save_path = save_path, 
                            verbose = 1) # Evaluate the model every 10000 timesteps and stop the training when the mean reward is greater than 500


if Train_with_callbacks:
    print('Training the agent with callbacks')
    # create a new PPO model and assign the callback
    model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log=log_path)
    model.learn(total_timesteps=20000, callback=eval_callback) # Train the model for 20000 timesteps (iterations) and assign the eval_callback



############################################
## Modify NN architecture (change policy) ##
############################################

net_arch = dict(pi = [128,128,128,128], vf = [128,128,128,128]) # Change the policy to a 4-layer NN with 128 neurons in each layer. pi = policy (actor), vf = value function (critic). net_arch is a dictionary. This dictionary has two keys: 'pi' and 'vf'. Each key maps to a list of four integers.
# see baseliane3 documentation for more information about custom policies

if Train_with_mod_NN:
    print('Training the agent with modified NN architecture and callbacks')
    model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log=log_path, policy_kwargs = {'net_arch':net_arch}) # Create a PPO model with the new architecture. OBS: `dict(key=value)` is just another way to create a dictionary. It's equivalent to `{'key': value}`. 
    model.learn(total_timesteps=20000, callback = eval_callback) 

##############################
## Use different algorithms ##
##############################

from stable_baselines3 import DQN

if Train_with_diff_alg:
    print('Training the agent with DQN algorithm')
    DQN_path = os.path.join('Training', 'Saved Models', 'DQN_Model_CartPole')
    model = DQN('MlpPolicy', env, verbose = 1, tensorboard_log=log_path) # Create a DQN model
    model.learn(total_timesteps=20000) # Train the model for 20000 timesteps (iterations)
    model.save(DQN_path)