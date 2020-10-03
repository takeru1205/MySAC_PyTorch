from time import time
import gym
import pybullet_envs
from torch.utils.tensorboard import SummaryWriter

from agent import SAC

# ENV_ID = 'InvertedPendulumBulletEnv-v0'
ENV_ID = 'HalfCheetahBulletEnv-v0'
# ENV_ID = 'HalfCheetah-v2'
env = gym.make(ENV_ID)
test_env = gym.make(ENV_ID)

seed = 0

env.seed(seed)
test_env.seed(2**31-seed)

writer = SummaryWriter(log_dir='logs/{}'.format(ENV_ID))

agent = SAC(env, entropy_tune=False, reward_scale=5.0, writer=writer)

initial_step = 10**4
eval_step = 10 ** 4
num_test = 3
train_steps = 10 ** 6

# Initial Step
state = env.reset()
episode_steps = 0
for tstep in range(initial_step):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    episode_steps += 1

    if episode_steps == env.spec.max_episode_steps:
        done_masked = False
    else:
        done_masked = done

    agent.store_transition(state, action, next_state, reward, done_masked)
    state = next_state

    if done:
        episode_steps = 0
        state = env.reset()

# Train Step
state = env.reset()
t1 = time()
cumulative_reward = 0
episode_steps = 0
for tstep in range(train_steps):
    action, _ = agent.get_action(state)
    next_state, reward, done, _ = env.step(action * env.action_space.high[0])
    episode_steps += 1
    cumulative_reward += reward
    if episode_steps == env.spec.max_episode_steps:
        done_masked = False
    else:
        done_masked = done

    agent.store_transition(state, action, next_state, reward, done_masked)
    loss_critic, loss_actor, loss_entropy = agent.update()
    if tstep % 1000 == 0:
        writer.add_scalar("loss/critic", loss_critic, tstep)
        writer.add_scalar("loss/actor", loss_actor, tstep)
        writer.add_scalar("loss/entropy", loss_entropy, tstep)

    state = next_state

    if done:
        writer.add_scalar("log/cumulative reward", cumulative_reward, tstep)
        state = env.reset()
        cumulative_reward = 0
        episode_steps = 0

    if tstep % eval_step == 0:
        test_score = []
        for _ in range(num_test):
            eval_state = test_env.reset()
            eval_cumulative_reward = 0
            eval_done = False
            while not eval_done:
                eval_action = agent.get_deterministic_action(eval_state)
                eval_next_state, eval_reward, eval_done, _ = test_env.step(eval_action * test_env.action_space.high[0])
                eval_cumulative_reward += eval_reward
                eval_state = eval_next_state
            test_score.append(eval_cumulative_reward)
        writer.add_scalar("log/Test Reward", sum(test_score) / num_test, tstep)
        t2 = time()
        elapsed_time = t2 - t1
        elapsed_hour = int(elapsed_time // 3600)
        elapsed_min = int((elapsed_time % 3600) // 60)
        elapsed_sec = int((elapsed_time % 3600) % 60)
        print('Num Steps:{:<6}, Evaluate {} average score {}, Time: {:02}:{:02}:{:02}'.format(
            tstep, num_test, sum(test_score) / num_test, elapsed_hour, elapsed_min, elapsed_sec
        ))


test_score = []
for _ in range(num_test):
    eval_state = test_env.reset()
    eval_cumulative_reward = 0
    eval_done = False
    while not eval_done:
        eval_action = agent.get_deterministic_action(eval_state)
        eval_next_state, eval_reward, eval_done, _ = test_env.step(eval_action * test_env.action_space.high[0])
        eval_cumulative_reward += eval_reward
        eval_state = eval_next_state
    test_score.append(eval_cumulative_reward)
writer.add_scalar("log/Test Reward", sum(test_score) / num_test, train_steps)
t2 = time()
elapsed_time = t2 - t1
elapsed_hour = int(elapsed_time // 3600)
elapsed_min = int((elapsed_time % 3600) // 60)
elapsed_sec = int((elapsed_time % 3600) % 60)
print('Num Steps:{:<6}, Evaluate 10 average score {}, Time: {:02}:{:02}:{:02}'.format(
    train_steps, sum(test_score) / num_test, elapsed_hour, elapsed_min, elapsed_sec
))

env.close()
