import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

class CustomBipedalWalkerEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # Transitional update: modify reward based on some condition
        if not done and reward < -100:  # Penalize the agent heavily if it's likely to fall
            reward -= 50  # Additional penalty
        return obs, reward, done, truncated, info

# Create the environment
def make_custom_env():
    env = gym.make("BipedalWalker-v3", render_mode="human")
    return CustomBipedalWalkerEnv(env)

env = make_custom_env()

# Add some noise for exploration
action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape[0]), sigma=0.1 * np.ones(env.action_space.shape[0]))

# Initialize TD3 model
model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the model
model.save("td3_bipedalwalker")

# Test the trained model
obs = env.reset()[0]
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, truncated, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()[0]

env.close()
