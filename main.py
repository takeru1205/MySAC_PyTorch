import gym
gym.logger.set_level(40)

env = gym.make('Pendulum-v0')


state = env.reset()
cumulative_reward = 0
for i in range(env.spec.max_episode_steps):
    action = env.action_space.sample()

    next_state, reward, done, info = env.step(action)

    cumulative_reward += reward
    
print(f'Reward: {cumulative_reward:>8.2f}')

