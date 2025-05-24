# test_env.py
from rl_turtlesim.env import TurtleEnv

env = TurtleEnv()
obs, _ = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, res = env.step(action)
    print(f"Obs: {obs}, Reward: {reward}, Caught: {res['turtles_caught']}")
