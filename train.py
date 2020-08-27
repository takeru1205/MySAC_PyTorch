import gym
import numpy as np
import torch

from agent import SAC

env = gym.make('Pendulum-v0')
epoch = 1000

# seed
np.random.seed(42)
env.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

agent = SAC(env)

all_timesteps = 0
start_steps = 1000
for e in range(epoch):
    state = env.reset()
    cumulative_reward = 0
    for i in range(env.spec.max_episode_steps):
        if all_timesteps <= start_steps:
            action = env.action_space.sample()
        else:
            action, _ = agent.get_action(state)
        state_, reward, done, _ = env.step(action * env.action_space.high[0])
        env.render()
        agent.store_transition(state, action, state_, reward, done)

        state = state_
        cumulative_reward += reward

        if all_timesteps > start_steps:
            agent.update(all_timesteps)
        all_timesteps += 1
    print('Epoch : {} / {}, Cumulative Reward : {}'.format(e, epoch, cumulative_reward))
    # writer.add_scalar("reward", cumulative_reward, e)

