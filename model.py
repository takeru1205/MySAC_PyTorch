import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import reparameterize


class Actor(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2 * action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).chunk(2, dim=-1)[0]
        x = torch.tanh(x)
        return x

    def reparameterize(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        means, log_stds = self.fc3(x).chunk(2, dim=-1)
        return reparameterize(means, log_stds.clamp_(-20, 2))


class Critic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        self.fc4 = nn.Linear(input_dim + action_dim, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)

    def forward(self, x, a):
        q1 = F.relu(self.fc1(torch.cat([x, a], dim=1)))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(torch.cat([x, a], dim=1)))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2
