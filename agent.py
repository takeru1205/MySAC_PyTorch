import torch
import torch.optim as optim
from model import Actor, Critic
from memory import ReplayMemory


class SAC(object):
    def __init__(self, env, writer=None, entropy_tune=True):
        self.env = env
        self.writer = writer

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.max_action = env.action_space.high[0]

        # Network
        self.actor = Actor(state_dim, action_dim).to('cuda')
        self.critic = Critic(state_dim, action_dim).to('cuda')
        self.target_critic = Critic(state_dim, action_dim).to('cuda')
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()
        # Invalid gradient
        for param in self.target_critic.parameters():
            param.requires_grad = False

        # Memory
        self.memory = ReplayMemory(state_dim, action_dim)

        # Parameters
        self.gamma = 0.99
        self.tau = 0.005
        self.reward_scale = 1.0

        # Entropy Tune
        self.entropy_tune = entropy_tune
        if entropy_tune:
            self.target_entropy = -torch.prod(
                torch.Tensor(env.action_space.shape).to('cuda')
            ).item()
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device='cuda')
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        else:
            self.alpha = 0.2

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float, device='cuda').unsqueeze_(0)
        with torch.no_grad():
            action, log_pi = self.actor.reparameterize(state)
        return action.cpu().numpy()[0], log_pi.item()

    def exploit(self, state):
        state = torch.tensor(state, dtype=torch.float, device='cuda').unsqueeze_(0)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]

    def store_transition(self, state, action, state_, reward, done):
        self.memory.store_transition(state, action, state_, reward, done)

    def update_target(self):
        for t, s in zip(self.target_critic.parameters(), self.critic.parameters()):
            t.data.mul_(1.0 - self.tau)
            t.data.add_(self.tau * s.data)

    def update(self, timesteps, batch_size=256):
        states, actions, states_, rewards, terminals = self.memory.sample(batch_size)

        critic_loss = self.update_critic(states, actions, states_, rewards, terminals)
        actor_loss, entropy, entropy_loss = self.update_actor(states)
        self.update_target()

        # tensorboard
        if timesteps % 100 == 0 and self.writer:
            self.writer.add_scalar("Loss/Critic", critic_loss, timesteps)
            self.writer.add_scalar("Loss/Actor", actor_loss, timesteps)
            if self.entropy_tune:
                self.writer.add_scalar("Loss/Entropy", entropy_loss, timesteps)

    def update_critic(self, states, actions, states_, rewards, terminals):
        current_q1, current_q2 = self.critic(states, actions)

        with torch.no_grad():
            next_actions, log_pis = self.actor.reparameterize(states_)
            next_q1, next_q2 = self.target_critic(states_, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha + log_pis
        target_q = rewards * self.reward_scale + terminals * self.gamma * next_q

        loss_critic1 = (current_q1 - target_q).pow_(2).mean()
        loss_critic2 = (current_q2 - target_q).pow_(2).mean()
        loss_critic = loss_critic1 + loss_critic2

        self.critic_optimizer.zero_grad()
        loss_critic.backward(retain_graph=False)
        self.critic_optimizer.step()

        return loss_critic.item()

    def update_actor(self, states):
        actions, log_pis = self.actor.reparameterize(states)
        q = torch.min(*self.critic(states, actions))
        loss_entropy = 0

        # entropy tuning
        if self.entropy_tune:
            loss_entropy = -(self.log_alpha * (log_pis + self.target_entropy).detach()).mean()
            self.alpha = self.log_alpha.exp()

        loss_actor = (self.alpha * log_pis - q).mean()

        self.actor_optimizer.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.actor_optimizer.step()

        if self.entropy_tune:
            self.alpha_optimizer.zero_grad()
            loss_entropy.backward(retain_graph=False)
            self.alpha_optimizer.step()

        return loss_actor.item(), log_pis, loss_entropy.item()

    def evaluate(self, evaluate_env, times=10):
        episode_returns = []
        for _ in range(times):
            state = evaluate_env.reset()
            done = False
            episode_return = 0.0

            while not done:
                action = self.exploit(state)
                state, reward, done, _ = evaluate_env.step(action)
                episode_return += reward
            episode_returns.append(episode_return)
        return sum(episode_returns) / times

    def save_model(self, path='models/'):
        torch.save(self.actor.state_dict(), path + 'actor')
        torch.save(self.critic.state_dict(), path + 'critic')

    def load_model(self, path='models/'):
        self.actor.load_state_dict(torch.load(path + 'actor'))
        self.critic.load_state_dict(torch.load(path + 'critic'))
