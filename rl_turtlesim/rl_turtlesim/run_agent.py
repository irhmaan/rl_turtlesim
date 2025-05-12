from stable_baselines3 import PPO
from rl_turtlesim.env import TurtleEnv

env = TurtleEnv()
model = PPO.load("ppo_turtle")

obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, status, res = env.step(action)
    print(f"Obs: {obs}, Reward: {reward}, Status: {done}, Caught: {res['turtles_caught']}")
