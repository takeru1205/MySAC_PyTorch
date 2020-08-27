import torch
import torch.optim as optim
from model import Actor, Critic
from memory import ReplayMemory

from const import *


class SAC(object):
    def __init__(self, env, writer=None):
        self.env = env
        self.writer = writer

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.max_action = env.action_space.high[0]

        # Network
        self.actor = Actor(state_dim, action_dim).to('cuda')
        self.critic = Critic(state_dim, action_dim).to('cuda')
        self.target_critic = Critic(state_dim, action_dim).to('cuda')
        self.target_critic.load_state_dict(self.critic.state_dict()).eval()
        # Invalid gradient
        for param in self.target_critic.parameters():
            param.requires_grad = False

        # Memory
        self.memory = ReplayMemory(state_dim, action_dim)

        # Parameters
        self.gamma = 0.99
        self.tau = 0.005
        self.hard_target_update_interval = 10000
        self.hard_target_update_gradient_step = 4
        self.alpha = 0.2  # temperature parameter
        self.reward_scale = 1.0

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=weight_decay)

    def get_action(self, state):
        with torch.no_grad():
            action, log_pi = self.actor.reparameterize(torch.from_numpy(state).float().to('cuda'))
        return action.detach().cpu().numpy(), log_pi.item()

    def store_transition(self, state, action, state_, reward, done):
        self.memory.store_transition(state, action, state_, reward, done)

    def update_target(self):
        for t, s in zip(self.target_critic.parameters(), self.critic.parameters()):
            t.data.mul_(1.0 - self.tau)
            t.data.add_(self.tau * s.data)

    def update(self, batch_size=256):
        states, actions, states_, rewards, terminals = self.memory.sample(batch_size)

        self.update_critic(states, actions, rewards, terminals, states_)
        self.update_actor(states)
        self.update_target()

        del states, actions, states_, rewards, terminals

    def update_critic(self, states, actions, rewards, dones, next_states):
        current_q1, current_q2 = self.critic(states, actions)

        with torch.no_grad():
            next_actions, log_pis = self.actor.reparameterize(next_states)
            next_q1, next_q2 = self.target_critic(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha + log_pis
        target_q = rewards * self.reward_scale + (1.0 - dones) * self.gamma * next_q

        loss_critic1 = (current_q1 - target_q).pow_(2).mean()
        loss_critic2 = (current_q2 - target_q).pow_(2).mean()
        loss_critic = loss_critic1 + loss_critic2

        self.critic_optimizer.zero_grad()
        loss_critic.backward(retain_graph=False)
        del loss_critic, loss_critic1, loss_critic2, target_q, current_q1, current_q2
        self.critic_optimizer.step()

    def update_actor(self, states):
        actions, log_pis = self.actor.reparameterize(states)
        q1, q2 = self.critic(states, actions)
        loss_actor = (self.alpha * log_pis - torch.min(q1, q2)).mean()

        self.actor_optimizer.zero_grad()
        loss_actor.backward(retain_graph=False)
        del loss_actor, q1, q2, actions, log_pis
        self.actor_optimizer.step()

    def save_model(self, path='models/'):
        torch.save(self.actor.state_dict(), path + 'actor')
        torch.save(self.critic.state_dict(), path + 'critic')

    def load_model(self, path='models/'):
        self.actor.load_state_dict(torch.load(path + 'actor'))
        self.critic.load_state_dict(torch.load(path + 'critic'))
