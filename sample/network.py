import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class BaseNet(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


def calculate_log_pi(log_stds, noises, actions):
    """ 確率論的な行動の確率密度を返す． """
    # ガウス分布 `N(0, stds * I)` における `noises * stds` の確率密度の対数(= \log \pi(u|a))を計算する．
    # (torch.distributions.Normalを使うと無駄な計算が生じるので，下記では直接計算しています．)
    gaussian_log_probs = \
        (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)

    # tanh による確率密度の変化を修正する．
    log_pis = gaussian_log_probs - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

    return log_pis


def reparameterize(means, log_stds):
    """ Reparameterization Trickを用いて，確率論的な行動とその確率密度を返す． """
    # 標準偏差．
    stds = log_stds.exp()
    # 標準ガウス分布から，ノイズをサンプリングする．
    noises = torch.randn_like(means)
    # Reparameterization Trickを用いて，N(means, stds)からのサンプルを計算する．
    us = means + noises * stds
    # tanh　を適用し，確率論的な行動を計算する．
    actions = torch.tanh(us)

    # 確率論的な行動の確率密度の対数を計算する．
    log_pis = calculate_log_pi(log_stds, noises, actions)

    return actions, log_pis, None


class Actor(BaseNet):
    def __init__(self, in_size, out_size, log_std_max=2, log_std_min=-20, eps=1e-6):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(in_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, out_size * 2)
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.eps = eps

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        mean, log_std = torch.chunk(self.fc3(x), chunks=2, dim=-1)
        return mean, log_std.clamp(self.log_std_min, self.log_std_max)

    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        z = Normal(mean, std)
        e = z.rsample()
        action = torch.tanh(e)
        log_pi = z.log_prob(e) - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return action, log_pi, torch.tanh(mean)  # probabilistic action, entropy, deterministic action


class QNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)
        return x


class Critic(BaseNet):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.q1 = QNet(state_size, action_size)
        self.q2 = QNet(state_size, action_size)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)

        q1 = self.q1(x)
        q2 = self.q2(x)

        return q1, q2
