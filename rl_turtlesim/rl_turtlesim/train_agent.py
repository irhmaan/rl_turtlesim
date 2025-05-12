from stable_baselines3 import PPO
from rl_turtlesim.env import TurtleEnv
from stable_baselines3.common.env_checker import check_env

env = TurtleEnv()
check_env(env)  # optional sanity check

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=30000, progress_bar=True)

model.save("ppo_turtle")
