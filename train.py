import gym
import pybullet_envs
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agent import SAC

# ENV_ID = 'Pendulum-v0'
ENV_ID = 'InvertedPendulumBulletEnv-v0'
env = gym.make(ENV_ID)
evaluate_env = gym.make(ENV_ID)
epoch = 5 * 10 ** 4

writer = SummaryWriter(log_dir='./logs/{}'.format(ENV_ID))

# seed
np.random.seed(42)
env.seed(42)
evaluate_env.seed(1234)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

agent = SAC(env, writer)

all_timesteps = 0
start_steps = 10000
evaluate_epoch = 100

for e in range(epoch):
    state = env.reset()
    cumulative_reward = 0
    for i in range(env.spec.max_episode_steps):
        if all_timesteps <= start_steps:
            action = env.action_space.sample()
        else:
            action, _ = agent.get_action(state)
        state_, reward, done, info = env.step(action * env.action_space.high[0])
        # env.render()
        if done and i == env.spec.max_episode_steps-1:
            agent.store_transition(state, action, state_, reward, False)
            agent.update(all_timesteps)
            all_timesteps += 1
            break
        agent.store_transition(state, action, state_, reward, done)

        state = state_
        cumulative_reward += reward

        if all_timesteps > max(start_steps, int(1e+4)):
            agent.update(all_timesteps)
        all_timesteps += 1
        if done:
            break
    if e % evaluate_epoch == 0:
        evaluate_reward = agent.evaluate(evaluate_env)
        print('Epoch : {} / {}, Means of Evaluate Reward : {}'.format(e, epoch, evaluate_reward))
        writer.add_scalar("Reward/Evaluate", evaluate_reward, e)
    writer.add_scalar('Reward/Train', cumulative_reward, e)
    # print('Epoch : {} / {}, Episode Steps: {}, Cumulative Reward : {}'.format(e, epoch, i, cumulative_reward))
