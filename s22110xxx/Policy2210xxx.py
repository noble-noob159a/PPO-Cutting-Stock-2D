from policy import Policy
import torch.nn as nn
import torch
from torch.optim import Adam
import time
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from module import *


def Policy2210xxx(Policy):
    pass


class PPO:
    def __init__(self, env):
        self.timesteps_per_batch = 3000  # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 600 # Max number of timesteps per episode
        self.n_updates_per_iteration = 15 # Number of times to update actor/critic per iteration
        self.lr = 0.0001  # Learning rate of actor optimizer
        self.gamma = 0.99  # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2  # Recommended 0.2, helps define the threshold to clip the ratio during SGA

        # Miscellaneous parameters
        self.render = False  # If we should render during rollout
        self.render_every_i = 10  # Only render every n iterations
        self.save_freq = 100  # How often we save in number of iterations
        self.seed = None

        # Extract environment information
        self.env = env
        self.customObs = []
        self.min_w = 50
        self.min_h = 50
        self.max_w = 100
        self.max_h = 100
        self.num_stocks = 100
        self.max_product_type = 25
        self.max_product_per_type = 20
        self.obs_dim = 2275  # 100*2(stocks) + 3*25(products) + 4*20*25(last moves)
        self.act_dim = [100,25,100,100]  # [stock idx, product idx, x, y]

        # Initialize actor and critic networks
        self.actor = PPOActor(self.obs_dim,self.act_dim)  # ALG STEP 1
        self.critic = ValueNet(self.obs_dim)

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,  # timesteps so far
            'i_so_far': 0,  # iterations so far
            'batch_lens': [],  # episodic lengths in batch
            'batch_rews': [],  # episodic returns in batch
            'actor_losses': [],  # losses of actor network in current iteration
        }

    def learn(self, total_timesteps):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far
        while t_so_far < total_timesteps:  # ALG STEP 2
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()  # ALG STEP 3

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Calculate advantage at k-th iteration
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()  # ALG STEP 5

            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach())

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './models/ppo_actor.pth')
                torch.save(self.critic.state_dict(), './models/ppo_critic.pth')

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        t = 0  # Keeps track of how many timesteps we've run so far this batch

        while t < self.timesteps_per_batch:
            ep_rews = []  # rewards collected per episode

            # Reset the environment. sNote that obs is short for observation.
            obs, _ = self.env.reset()
            self.resetCustomObservation(obs)
            done = False

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):
                # If render is specified, render the environment
                #print("Rolling out in",ep_t)
                if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
                    self.env.render()

                t += 1  # Increment timesteps ran this batch so far

                # Track observations in this batch
                batch_obs.append(self.customObs)

                # Calculate action and make a step in the env.
                # Note that rew is short for reward.
                action, log_prob = self.get_action(self.customObs)
                customReward = self.getCustomReward(obs,action)
                convertedAction = self.convertAction(action)
                obs, rew, terminated, truncated, _ = self.env.step(convertedAction)
                self.updateCustomObservation(obs,action)
                # Don't really care about the difference between terminated or truncated in this, so just combine them
                done = terminated | truncated

                # Track recent reward, action, and action log probability
                ep_rews.append(customReward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # If the environment tells us the episode is terminated, break
                if done:
                    break

            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        #print(batch_log_probs)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)  # ALG STEP 4

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):

        batch_rtgs = []
        # Iterate through each episode
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0  # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def get_action(self, obs):
        # Query the actor network for a mean action
        logits = self.actor(obs)
        probs = [F.softmax(logit, dim=-1) for logit in logits]
        actions = []
        log_probs = torch.tensor(0, dtype=torch.float)
        for prob in probs:
            dist = Categorical(prob)
            action = dist.sample()
            log_probs += dist.log_prob(action)
            actions.append(action.item())

        return np.array(actions), log_probs.detach()

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()

        logits = self.actor(batch_obs)
        probs = [F.softmax(logit, dim=-1) for logit in logits]
        log_probs = [0 for _ in range(batch_obs.shape[0])]
        log_probs = torch.tensor(log_probs, dtype=torch.float)
        batch_acts = torch.transpose(batch_acts, 0, 1)
        for prob, acts in zip(probs, batch_acts):
            dist = Categorical(prob)
            log_probs += dist.log_prob(acts)
        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs

    def _log_summary(self):
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []

    def resetCustomObservation(self, obs):
        stocksInfo = []
        for stock in obs["stocks"]:
            width = self.min_w
            for i in range(self.min_w, self.max_w):
                if stock[i, 0] == -1:
                    width += 1
                else:
                    break
            height = self.min_h
            for i in range(self.min_h, self.max_h):
                if stock[0, i] == -1:
                    height += 1
                else:
                    break
            stocksInfo.append(width)
            stocksInfo.append(height)

        productInfo = []
        for product in obs["products"]:
            productInfo.append(int(product["size"][0]))
            productInfo.append(int(product["size"][1]))
            productInfo.append(product["quantity"])
        for _ in range(self.max_product_type - len(obs["products"])):
            productInfo += [-1, -1, -1]
        lastMoves = []
        for _ in range(self.max_product_per_type*self.max_product_type):
            lastMoves += [-1, -1, -1, -1]
        self.customObs = stocksInfo + productInfo + lastMoves

    def updateCustomObservation(self, newObs, action):
        width, height = self.customObs[200 + action[1]*3], self.customObs[201 + action[1]*3]
        if width <= 0 or height <= 0:
            return
        update = False
        for i, product in enumerate(newObs["products"]):
            if int(product["size"][0]) == width and int(product["size"][1]) == height:
                prevQuantity = self.customObs[202+i*3]
                newQuantity = product["quantity"]
                if prevQuantity != newQuantity:
                    update = True
                    self.customObs[202 + i * 3] = newQuantity
                    break
        if not update:
            return
        start, step = 275, 4
        for i in range(500):
            if self.customObs[start+i*step] == -1:
                self.customObs[start + i * step:start + step + i * step] = action
                break

    def convertAction(self, action):
        width, height = self.customObs[200 + action[1]*3], self.customObs[201 + action[1]*3]
        return {"stock_idx": action[0], "size": np.array([width,height]), "position": (action[2], action[3])}

    def getCustomReward(self,obs,action):
        width, height = self.customObs[200 + action[1] * 3], self.customObs[201 + action[1] * 3]
        quantity = self.customObs[202 + action[1] * 3]
        stockIdx, x, y = action[0], action[2], action[3]
        if width <= 0 or height <= 0 or quantity <= 0:
            return -10
        stock = obs["stocks"][stockIdx]
        stockWidth, stockHeight = self.customObs[stockIdx*2], self.customObs[1 + stockIdx*2]
        if x + width > stockWidth or y + height > stockHeight:
            return -5
        if not (np.all(stock[x: x + width, y: y + height] == -1)):
            return -5
        if stockIdx > 0:
            lastWaste = np.count_nonzero(obs["stocks"][stockIdx-1] == -1)
            if lastWaste >= width*height:
                return -10
        if x > 0:
            if stock[x-1, y] == -1:
                return -1
        if y > 0:
            if stock[x, y-1] == -1:
                return -1
        if x > 0 and y > 0:
            if stock[x-1, y-1] == -1:
                return -1
        remainQuantity = 0
        for i in range(25):
            q = self.customObs[202 + i*3]
            if q > 0: remainQuantity += q
        if remainQuantity == 0: return 100
        return 10

