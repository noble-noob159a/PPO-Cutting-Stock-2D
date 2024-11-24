import gymnasium as gym
from Policy2210xxx import *
import gym_cutting_stock

env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)

agent = PPO()
observation, info = env.reset(seed=42)
#agent.train(total_timestep=50_000)
ep = 0
while ep < 1:
    action = agent.get_action(observation, info)
    observation, reward, terminated, truncated, info = env.step(action)
    # print(observation["stocks"])
    if terminated or truncated:
        print(info)
        observation, info = env.reset(seed=ep)
        ep += 1
