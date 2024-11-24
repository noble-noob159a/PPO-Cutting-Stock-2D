import torch
from module import *
import numpy as np
import matplotlib.pyplot as plt
from policy import Policy


def Policy2210xxx(Policy):
    pass


class PPO:
    def __init__(self, env=None, load_check_pontis=False):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.gamma = 0.99
        self.policy_clip = 0.2
        self.n_epochs = 15
        self.gae_lambda = 0.95
        self.rollout_steps = 50  # N - big batch
        self.batch_size = 10
        self.leaning_rate = 0.1

        self.customObs = []
        self.obsInfo = dict()
        self.filter_out = 256
        self.n_actions = 3  # [stock idx, x, y]
        self.actor = ActorNetwork(self.n_actions, self.filter_out, self.leaning_rate)
        self.env = env
        if load_check_pontis or env is None:
            self.actor.load_checkpoint()
            # self.critic.load_checkpoint()
        self.memory = PPOMemory(self.batch_size)
        self.old_action = None  # for inference

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        # self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        # self.critic.load_checkpoint()

    def choose_action(self, observation):
        # state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)
        dist, value = self.actor([observation])
        action = dist.sample()
        probs = torch.squeeze(dist.log_prob(action).sum()).item()
        value = torch.squeeze(value).item()
        return action, probs, value

    def get_action(self, obs, info):
        if len(self.customObs) == 0:
            self.resetCustomObservation(obs)
            self.stepInfo()
        else:
            self.updateCustomObservation(obs, self.old_action)
        with torch.no_grad():
            #state = torch.tensor([self.customObs], dtype=torch.float).to(self.actor.device)
            dist, val = self.actor([self.customObs])
            action = dist.sample()
        scaledAction = self.getScaledAction(action)
        convertedAction = self.convertAction(scaledAction)
        reward, done = self.getCustomReward(obs, scaledAction)
        self.old_action = scaledAction
        if done:
            self.customObs = []
        return convertedAction

    def stepInfo(self, postInfo=False):
        count = quantity = 0
        for p in self.obsInfo["productInfo"]:
            count += 1
            quantity += p[3]

        if postInfo:
            print(f'Remaining products: {quantity}, intactStock: {len(self.obsInfo["intact"])}')
        else:
            print(f'---------------------------------------',
                  f'\nTotal product types: {count}, products demand: {quantity}')

    def train(self, total_timestep, timestep_per_game=2_000):
        figure_file = 'rewards.png'
        best_score = -10000
        score_history = []
        learn_iters = 0
        avg_score = 0
        n_steps = 0

        while n_steps < total_timestep:
            observation, _ = self.env.reset()
            self.resetCustomObservation(observation)
            #
            done = False
            score = 0
            game_timestep = 0
            self.stepInfo()
            while game_timestep < timestep_per_game and (not done):
                action, prob, val = self.choose_action(self.customObs)
                scaledAction = self.getScaledAction(action)
                convertedAction = self.convertAction(scaledAction)
                reward, _ = self.getCustomReward(observation, scaledAction)
                observation_, _, done, _, info = self.env.step(convertedAction)

                n_steps += 1
                game_timestep += 1
                score += reward
                self.remember(self.customObs, action, prob, val, reward, done)
                self.updateCustomObservation(observation_, scaledAction)
                if n_steps % self.rollout_steps == 0:
                    self.learn()
                    learn_iters += 1
                observation = observation_
                if n_steps % 1000 == 0:
                    print("done step ", n_steps)
                    self.stepInfo(True)
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            normalized_reward = score / game_timestep
            if normalized_reward > best_score:
                best_score = normalized_reward
                self.save_models()

            self.stepInfo(True)
            print('Game step', game_timestep, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                  'time_steps', n_steps, 'learning_steps', learn_iters)
        self.save_models()
        x = [i + 1 for i in range(len(score_history))]
        self.plot_learning_curve(x, score_history, figure_file)

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
                reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)

            values = torch.tensor(values).to(self.actor.device)
            for batch in batches:
                #states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                dist, critic_value = self.actor(state_arr[batch])

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions).sum()
                prob_ratio = new_probs.exp() / old_probs.exp()
                # prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                     1 + self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()

        self.memory.clear_memory()

    @staticmethod
    def plot_learning_curve(x, scores, figure_file):
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
        plt.plot(x, running_avg)
        plt.title('Running average of previous 100 scores')
        plt.savefig(figure_file)

    def getScaledAction(self, action):
        act = action.squeeze(dim=0)[0].cpu()
        pr_idx = torch.round(torch.tanh(act) + 1).int().item()
        w, h = self.obsInfo["chosenStock"][pr_idx][1], self.obsInfo["chosenStock"][pr_idx][2]
        action_max = torch.tensor([2, w - 1, h - 1], dtype=torch.int).to(self.device)
        action_min = torch.tensor([0, 0, 0], dtype=torch.int).to(self.device)
        bounded_act = torch.tanh(action)
        scaled_act = action_min + (bounded_act + 1) * 0.5 * (action_max - action_min)
        return torch.round(scaled_act).int()

    def initCustomObs(self, obs):
        self.customObs = []
        chosen_product = []  # idx, w, h, quantity
        product_area = 0
        for product in self.obsInfo["productInfo"]:
            if product[3] > 0:
                chosen_product = [product[0], product[1], product[2], product[3]]
                product_area = product[1] * product[2]
                break
        if len(chosen_product) == 0:
            chosen_product = [0, 0, 0, 0]
        chosen_stock = []  # 3 stocks - [i,w,h]
        for stock in self.obsInfo["stockInfo"]:
            w, h = stock[1:3]
            empty_area = np.count_nonzero(obs["stocks"][stock[0]] == -1)
            if empty_area >= product_area:
                self.customObs.append(obs["stocks"][stock[0]][:w, :h])
                chosen_stock.append([stock[0], w, h])
            if len(chosen_stock) >= 3:
                break
        if chosen_stock[-1] not in self.obsInfo["intact"]:
            for i in self.obsInfo["intact"]:
                area = np.count_nonzero(obs["stocks"][i] == -1)
                if i not in chosen_stock and area >= product_area:
                    w = np.count_nonzero(obs["stocks"][i][:, 0] > -2)
                    h = np.count_nonzero(obs["stocks"][i][0, :] > -2)
                    self.customObs[-1] = obs["stocks"][i][:w, :h]
                    chosen_stock[-1] = [i, w, h]
                    break
        while len(chosen_stock) < 3:
            chosen_stock.append([0, 0, 0])
        self.customObs.append([chosen_product[1], chosen_product[2]])
        self.obsInfo["chosenStock"] = chosen_stock
        self.obsInfo["chosenProduct"] = chosen_product

    def resetCustomObservation(self, obs):
        stocksInfo = []  # idx, w, h
        for i, stock in enumerate(obs["stocks"]):
            w = np.count_nonzero(stock[:, 0] > -2)
            h = np.count_nonzero(stock[0, :] > -2)
            stocksInfo.append([i, w, h])
        productInfo = []  # idx, w, h, quantity
        for i, product in enumerate(obs["products"]):
            productInfo.append([i, int(product["size"][0]), int(product["size"][1]), product["quantity"]])
        sorted_stock = sorted(stocksInfo, key=lambda x: x[1] * x[2])
        sorted_product = sorted(productInfo, key=lambda x: x[1] * x[2], reverse=True)
        intactStock = [stock[0] for stock in sorted_stock]
        self.obsInfo = {"stockInfo": sorted_stock, "productInfo": sorted_product, "stockCount": len(sorted_stock)
            , "productCount": len(sorted_product), "intact": intactStock, "chosenStock": [], "chosenProduct": []}
        self.initCustomObs(obs)

    def updateCustomObservation(self, newObs, action):
        action = np.array(action.squeeze(dim=0).cpu())
        productIdx = self.obsInfo["chosenProduct"][0]
        stockIdx = self.obsInfo["chosenStock"][action[0]][0]
        update = False
        for i, product in enumerate(newObs["products"]):
            if i == productIdx:
                prevQuantity = self.obsInfo["chosenProduct"][3]
                newQuantity = product["quantity"]
                if prevQuantity > newQuantity:
                    update = True
                    break
        if not update:
            return
        for i, product in enumerate(self.obsInfo["productInfo"]):
            if product[0] == productIdx:
                self.obsInfo["productInfo"][i][3] -= 1
                break
        if stockIdx in self.obsInfo["intact"]:
            self.obsInfo["intact"].remove(stockIdx)
        self.initCustomObs(newObs)

    def convertAction(self, action):
        action = np.array(action.squeeze(dim=0).cpu())
        stockIdx = self.obsInfo["chosenStock"][action[0]][0]
        width, height = self.obsInfo["chosenProduct"][1], self.obsInfo["chosenProduct"][2]
        return {"stock_idx": stockIdx, "size": np.array([width, height]), "position": (action[1], action[2])}

    def getCustomReward(self, obs, action):
        action = np.array(action.squeeze(dim=0).cpu())
        width, height = self.obsInfo["chosenProduct"][1], self.obsInfo["chosenProduct"][2]
        stockIdx, x, y = self.obsInfo["chosenStock"][action[0]][0], action[1], action[2]
        stock = obs["stocks"][stockIdx]
        stockWidth, stockHeight = self.obsInfo["chosenStock"][action[0]][1:]
        if x + width > stockWidth or y + height > stockHeight:
            return -1, False
        if not (np.all(stock[x: x + width, y: y + height] == -1)):
            return -1, False
        reward = 5
        if x > 0:
            if stock[x - 1, y] == -1:
                reward = -0.1
        if y > 0:
            if stock[x, y - 1] == -1:
                reward = -0.1
        if x > 0 and y > 0:
            if stock[x - 1, y - 1] == -1:
                reward = -0.1
        if action[0] > 0:
            idx = self.obsInfo["chosenStock"][action[0] - 1][0]
            lastWaste = np.count_nonzero(obs["stocks"][idx] == -1)
            if lastWaste >= width * height:
                reward = -0.3
        remain = 0
        done = False
        for product in obs["products"]:
            remain += product["quantity"]
        if remain == 1:
            # print('done')
            done = True
            reward += 10
        return reward, done
