import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOActor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 32)
        self.heads = nn.ModuleList([nn.Linear(32, dim) for dim in out_dim])
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64)
        self.norm3 = nn.LayerNorm(32)

    def forward(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        x = F.relu(self.layer1(obs))
        x_shortcut = x
        x = self.norm1(x)
        x = F.relu(self.layer2(x) + x_shortcut)
        x = self.norm2(x)
        x = F.relu(self.layer3(x))
        x = self.norm3(x)
        logits = [head(x) for head in self.heads]
        return logits


class ValueNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(32)

    def forward(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        x = F.relu(self.layer1(obs))
        x = self.norm1(x)
        x = F.relu(self.layer2(x))
        x = self.norm2(x)
        x = F.relu(self.layer3(x))
        return x
