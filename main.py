import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from s22110xxx.policy2210xxx import Policy2210xxx
import pandas as pd
import numpy as np
import time
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    #render_mode="human",  # Comment this line to disable rendering
)
def getFitness(obs):
    product_area = 0
    stock_area = 0
    for stock in obs["stocks"]:
        if (stock > -1).sum() > 0:
            w = np.count_nonzero(stock[:, 0] > -2)
            h = np.count_nonzero(stock[0, :] > -2)
            stock_area += w*h
            product_area += (stock > -1).sum()
    #print(product_area/stock_area)
    return product_area/stock_area

def getWasteRate(obs):
    product_area = 0
    waste_area = 0
    for stock in obs["stocks"]:
        if (stock > -1).sum() > 0:
            waste_area += (stock == -1).sum()
            product_area += (stock > -1).sum()
    return waste_area/product_area

if __name__ == "__main__":
    seeds = [42,52,62,72,82]
    df = pd.DataFrame(columns=["Best waste rate","Avg waste rate","Best trim loss","Avg trim loss","Best fitness","Avg fitness","Best time","Avg time"])
    for i, seed in enumerate(seeds):
        print("\nstart batch: ",i+1)
        waste_rate = []
        trim_loss = []
        m_time = []
        fitness = []
        observation, info = env.reset(seed=seed)
        agent = Policy2210xxx(1)
        ep = seed
        start_time = time.time()
        while ep < seed + 10:
            action = agent.get_action(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                ep += 1
                end_time = time.time()
                elapsed_time = end_time - start_time
                trim_loss.append(info["trim_loss"])
                fitness.append(getFitness(observation))
                waste_rate.append(getWasteRate(observation))
                m_time.append(elapsed_time)
                print(info, f'time: {elapsed_time} s')
                observation, info = env.reset(seed=ep)
                start_time = time.time()
        df.loc[i] = [np.min(waste_rate),np.mean(waste_rate),np.min(trim_loss),np.mean(trim_loss),np.max(fitness),np.mean(fitness),np.min(m_time),np.mean(m_time)]
    df.index = range(1, len(df) + 1)
    print(df)
    df.to_csv('heuristic.csv')
env.close()
