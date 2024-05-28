import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

# Define a custom wrapper for the LunarLander environment
# STL task: 
class CustomMountainCarEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        print(env.observation_space)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        # Optionally modify observation, reward, done here if needed
        return observation, reward, done, truncated, info

# Initialize the environment with the custom wrapper
def make_custom_env():
    env = gym.make("MountainCar-v0", render_mode="human")
    return CustomMountainCarEnv(env)

env = make_custom_env()

# Create the DQN model
model = DQN("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=50000)  # Adjust timesteps as needed

# Save the trained model
model.save("dqn_mountaincar")

# Load the model
model = DQN.load("dqn_mountaincar", env)

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Visualize the trained agent's performance
obs = env.reset()[0]
for _ in range(2000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, truncated, info = env.step(action)
    # print(">>> obs = ",obs)
    env.render()
    if done:
        obs = env.reset()[0]

env.close()
