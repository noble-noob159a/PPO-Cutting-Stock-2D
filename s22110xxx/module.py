import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class FilterCNN(nn.Module):
    def __init__(self, out_dim=256):
        super(FilterCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global average pooling to handle dynamic input sizes
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers for fixed-size output
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(),
        )
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.conv_layers(x)  # Pass through convolutional layers
        x = self.global_pool(x)  # Apply global average pooling
        x = self.fc_layers(x)  # Fully connected layer for embedding
        return x


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, filter_out, alpha, chkpt_dir='./models'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(filter_out*3 + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(filter_out*3 + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.filter = FilterCNN(filter_out)

        self.log_std = nn.Parameter(torch.zeros(n_actions, requires_grad=True, dtype=torch.float32), requires_grad=True)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def extractState(self,state):
        stock1 = []
        stock2 = []
        stock3 = []
        pro_size = []
        for s in state:
            stock1.append(torch.tensor(s[0],dtype=torch.float).unsqueeze(dim=0).unsqueeze(dim=0).to(self.device))
            stock2.append(torch.tensor(s[1],dtype=torch.float).unsqueeze(dim=0).unsqueeze(dim=0).to(self.device))
            stock3.append(torch.tensor(s[2],dtype=torch.float).unsqueeze(dim=0).unsqueeze(dim=0).to(self.device))
            pro_size.append(torch.tensor(s[3],dtype=torch.float).to(self.device))
            #print(stock1[0].shape)
        return [stock1,stock2,stock3],pro_size

    def forward(self, state):
        stocks, pro_size = self.extractState(state)
        state = torch.tensor([],dtype=torch.float).to(self.device)
        for i in range(len(pro_size)):
            s1,s2,s3 = stocks[0][i],stocks[1][i],stocks[2][i]
            f1,f2,f3 = self.filter(s1).squeeze(dim=0),self.filter(s2).squeeze(dim=0),self.filter(s3).squeeze(dim=0)
            #print(f1.shape,f2.shape,f3.shape,pro_size[i].shape)
            batch = torch.cat([f1,f2,f3,pro_size[i]],dim=0)
            state = torch.cat([state,batch.unsqueeze(dim=0)],dim=0)
        #print(state.shape)
        dist = self.actor(state)
        sigma = torch.exp(self.log_std)
        dist = Normal(dist, sigma)
        return dist, self.critic(state)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, chkpt_dir='./models'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(input_dims, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states,dtype=object), \
            np.array(self.actions), \
            np.array(self.probs), \
            np.array(self.vals), \
            np.array(self.rewards), \
            np.array(self.dones), \
            batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action.cpu())
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
