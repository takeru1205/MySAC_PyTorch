import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2 * action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x).chunk(2, dim=-1)[0]
        x = torch.tanh(x)
        return x

    def sample(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        means, log_stds = self.fc3(x).chunk(2, dim=-1)
        return utils.reparameterize(means, log_stds.clamp_(-20, 2))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim+action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        self.fc4 = nn.Linear(state_dim + action_dim, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)

    def forward(self, state, action):
        input_tensor = torch.cat([state, action], dim=-1)
        q1 = F.relu(self.fc1(input_tensor), inplace=True)
        q1 = F.relu(self.fc2(q1), inplace=True)
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(input_tensor), inplace=True)
        q2 = F.relu(self.fc5(q2), inplace=True)
        q2 = self.fc6(q2)

        return q1, q2
