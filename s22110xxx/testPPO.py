import gymnasium as gym
import sys
import torch
from Policy2210xxx import PPO
import gym_cutting_stock


def train(env, actor_model, critic_model):
    print(f"Training", flush=True)

    model = PPO(env=env)
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model,weights_only=True))
        model.critic.load_state_dict(torch.load(critic_model,weights_only=True))
        print(f"Successfully loaded.", flush=True)
    else:
        print(f"Training from scratch.", flush=True)
    model.learn(total_timesteps=200_000_000)


env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    #render_mode="human",  # Comment this line to disable rendering
)

actorDir = './models/ppo_actor.pth'
criticDir ='./models/ppo_critic.pth'
train(env, actorDir, criticDir)
