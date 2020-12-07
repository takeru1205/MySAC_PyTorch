import argparse
import gym
import numpy as np
import pybullet_envs
from torch.utils.tensorboard import SummaryWriter

from agent import SAC

parser = argparse.ArgumentParser()
parser.add_argument('--env', default='Pendulum-v0')
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--start_step', type=int, default=50000, help='Number of steps for uniform-random action selection')
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--update_after', type=int, default=1000, help='Number of interactions to collect before starting')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--reward_scale', type=float, default=1.)
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--evaluate_freq', type=int, default=100)
args = parser.parse_args()

epoch = 1000
env = gym.make(args.env)

writer = SummaryWriter(log_dir=f'logs/{args.env}', filename_suffix=args.env)
agent = SAC(env, writer=writer, reward_scale=1.0, lr=args.lr)

# Initial Random Actions
for _ in range(args.start_step // env.spec.max_episode_steps):
    state = env.reset()
    for _ in range(env.spec.max_episode_steps):
        action = env.action_space.sample()
        next_state, reward, terminal, _ = env.step(action)
        agent.store_step(state, action, next_state, reward, terminal)
        state = next_state

print('Train Start')
all_timesteps = 0
for e in range(epoch):
    state = env.reset()
    cumulative_reward = 0
    for s in range(env.spec.max_episode_steps):
        all_timesteps += 1
        action = agent.explore(state)
        next_state, reward, terminal, _ = env.step(action)

        if s == env.spec.max_episode_steps:
            terminal = 0
        else:
            terminal = terminal

        agent.store_step(state, action, next_state, reward, np.array(terminal))
        state = next_state
        cumulative_reward += reward

        agent.train(all_timesteps, args.batch_size)

        if terminal:
            break

    writer.add_scalar('log/reward', cumulative_reward, e)
    print(f'Epoch: {e}, Reward: {cumulative_reward}')

    if e % args.evaluate_freq == 0:
        cumulative_reward = 0
        state = env.reset()
        for s in range(env.spec.max_episode_steps):
            action = agent.exploit(state)
            next_state, reward, terminal, _ = env.step(action)
            state = next_state
            cumulative_reward += reward
            if terminal:
                break
        print(f'Evaluate: {e}, Reward: {cumulative_reward}')
        writer.add_scalar('log/Evaluate', cumulative_reward, e)

agent.save_weights('models')

env.close()
