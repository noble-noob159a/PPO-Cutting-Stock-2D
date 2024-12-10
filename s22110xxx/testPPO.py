import gymnasium as gym
from heuristic_policy import CPolicy
from ppo_policy import PPO
import gym_cutting_stock


test_mode = True
load_check_point = True


render = "human" if test_mode else None
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode=render,  # Comment this line to disable rendering
)
train_env = None if test_mode else env
agent = PPO(train_env,load_check_point)
if not test_mode:
    agent.train(total_timestep=100_000)
else:
    observation, info = env.reset(seed=42)
    ep = 35
    while ep < 30:
        action = agent.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        # print(observation["stocks"])
        if terminated or truncated:
            print(info)
            observation, info = env.reset(seed=ep)
            ep += 1
    print("agent2 result: ")
    agent = CPolicy()
    observation, info = env.reset(seed=42)
    ep = 0
    while ep < 10:
        action = agent.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        # print(observation["stocks"])
        if terminated or truncated:
            print(info)
            observation, info = env.reset(seed=ep)
            ep += 1
