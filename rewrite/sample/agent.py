import os
import numpy as np
import torch
import torch.optim as optim
from collections import namedtuple

from buffer import ReplayMemory
from network import Actor, Critic


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminal'))


def to_tensor_unsqueeze(ndarray):
    tensor = torch.from_numpy(ndarray).to(torch.float)
    return torch.unsqueeze(tensor, dim=0)


def update_params(optimizer, network, loss, grad_clip=None, retain_graph=False):
    optimizer.zero_grad()
    loss.backward(retain_graph=False)
    if grad_clip is not None:
        for p in network.modules():
            torch.nn.utils.clip_grad_norm_(p.parameters(), grad_clip)
    optimizer.step()


class SAC:
    def __init__(self, env, lr=3e-4, gamma=0.99, polyak=5e-3, alpha=0.2, reward_scale=1.0, cuda=True, writer=None):
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)
        self.target_critic = Critic(state_size, action_size).eval()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.critic.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.critic.q2.parameters(), lr=lr)

        self.target_critic.load_state_dict(self.critic.state_dict())
        for param in self.target_critic.parameters():
            param.requires_grad = False

        self.memory = ReplayMemory()

        self.gamma = gamma
        self.alpha = alpha
        self.polyak = polyak  # Always between 0 and 1, usually close to 1
        self.reward_scale = reward_scale

        self.writer = writer

        self.cuda = cuda
        if cuda:
            self.actor = self.actor.to('cuda')
            self.critic = self.critic.to('cuda')
            self.target_critic = self.target_critic.to('cuda')

    def explore(self, state):
        if self.cuda:
            state = torch.tensor(state).unsqueeze(0).to('cuda', torch.float)
        action, _, _ = self.actor.sample(state)
        # action, _ = self.actor(state)
        return action.cpu().detach().numpy().reshape(-1)

    def exploit(self, state):
        if self.cuda:
            state = torch.tensor(state).unsqueeze(0).to('cuda', torch.float)
        _, _, action = self.actor.sample(state)
        return action.cpu().detach().numpy().reshape(-1)

    def store_step(self, state, action, next_state, reward, terminal):
        state = to_tensor_unsqueeze(state)
        if action.dtype == np.float32:
            action = torch.from_numpy(action)
        next_state = to_tensor_unsqueeze(next_state)
        reward = torch.from_numpy(np.array([reward]).astype(np.float))
        terminal = torch.from_numpy(np.array([terminal]).astype(np.uint8))
        self.memory.push(state, action, next_state, reward, terminal)

    def target_update(self, target_net, net):
        for t, s in zip(target_net.parameters(), net.parameters()):
            # t.data.copy_(t.data * (1.0 - self.polyak) + s.data * self.polyak)
            t.data.mul_(1.0 - self.polyak)
            t.data.add_(self.polyak * s.data)

    def calc_target_q(self, next_states, rewards, terminals):
        with torch.no_grad():
            next_action, entropy, _ = self.actor.sample(next_states)  # penalty term
            next_q1, next_q2 = self.target_critic(next_states, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * entropy
        target_q = rewards * self.reward_scale + (1. - terminals) * self.gamma * next_q
        return target_q

    def calc_critic_loss(self, states, actions, next_states, rewards, terminals):
        q1, q2 = self.critic(states, actions)
        target_q = self.calc_target_q(next_states, rewards, terminals)

        q1_loss = torch.mean((q1 - target_q).pow(2))
        q2_loss = torch.mean((q2 - target_q).pow(2))
        return q1_loss, q2_loss

    def calc_actor_loss(self, states):
        action, entropy, _ = self.actor.sample(states)

        q1, q2 = self.critic(states, action)
        q = torch.min(q1, q2)

        # actor_loss = torch.mean(-q - self.alpha * entropy)
        actor_loss = (self.alpha * entropy - q).mean()
        return actor_loss, entropy

    def train(self, timestep, batch_size=256):
        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        transitions = Transition(*zip(*transitions))
        if self.cuda:
            states = torch.cat(transitions.state).to('cuda')
            actions = torch.stack(transitions.action).to('cuda')
            next_states = torch.cat(transitions.next_state).to('cuda')
            rewards = torch.stack(transitions.reward).to('cuda')
            terminals = torch.stack(transitions.terminal).to('cuda')
        else:
            states = torch.cat(transitions.state)
            actions = torch.stack(transitions.action)
            next_states = torch.cat(transitions.next_state)
            rewards = torch.stack(transitions.reward)
            terminals = torch.stack(transitions.terminal)
        # Compute target Q func
        q1_loss, q2_loss = self.calc_critic_loss(states, actions, next_states, rewards, terminals)
        # Compute actor loss
        actor_loss, mean = self.calc_actor_loss(states)
        update_params(self.q1_optimizer, self.critic.q1, q1_loss)
        update_params(self.q2_optimizer, self.critic.q2, q2_loss)
        update_params(self.actor_optimizer, self.actor, actor_loss)
        # target update
        self.target_update(self.target_critic, self.critic)

        if timestep % 100 and self.writer:
            self.writer.add_scalar('Loss/Actor', actor_loss.item(), timestep)
            self.writer.add_scalar('Loss/Critic', q1_loss.item(), timestep)

    def save_weights(self, path):
        self.actor.save(os.path.join(path, 'actor'))
        self.critic.save(os.path.join(path, 'critic'))
