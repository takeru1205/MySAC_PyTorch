import gym
gym.logger.set_level(40)

num_episode = 5

env = gym.make('Pendulum-v0')

for e in range(num_episode):
    cumulative_reward = 0
    state = env.reset()
    for i in range(env.spec.max_episode_steps):
        action = env.action_space.sample()

        next_state, reward, done, info = env.step(action)

        cumulative_reward += reward
        
    print(f'Episode: {e:>3}, Reward: {cumulative_reward:>8.2f}')

