from time import time
import gym
import pybullet_envs
from torch.utils.tensorboard import SummaryWriter

from agent import SAC

ENV_ID = 'InvertedPendulumBulletEnv-v0'
# ENV_ID = 'HalfCheetahBulletEnv-v0'
env = gym.make(ENV_ID)
test_env = gym.make(ENV_ID)

writer = SummaryWriter(log_dir='logs/{}_autotune'.format(ENV_ID))

agent = SAC(env, reward_scale=5.0, writer=writer)

initial_step = int(1e+4)
epoch = 3000
eval_step = 10000
num_test = 5
train_steps = 10 ** 6

# Initial Step
state = env.reset()
episode_steps = 0
for step in range(1, initial_step+1):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)

    if episode_steps == env.spec.max_episode_steps - 1:
        done_masked = False
    else:
        done_masked = done

    agent.store_transition(state, action, next_state, reward, done_masked)
    episode_steps += 1
    state = next_state

    if done:
        episode_steps = 0
        state = env.reset()

# Train Step
state = env.reset()
t1 = time()
cumulative_reward = 0
episode_steps = 0
for step in range(train_steps):
    action, _ = agent.get_action(state)
    next_state, reward, done, _ = env.step(action * env.action_space.high[0])
    cumulative_reward += reward
    if episode_steps == env.spec.max_episode_steps - 1:
        done_masked = False
    else:
        done_masked = done

    agent.store_transition(state, action, next_state, reward, done_masked)
    loss_critic, loss_actor, loss_entropy = agent.update()
    if step % 1000 == 0:
        writer.add_scalar("loss/critic", loss_critic, step)
        writer.add_scalar("loss/actor", loss_actor, step)
        writer.add_scalar("loss/entropy", loss_entropy, step)

    state = next_state

    if done:
        state = env.reset()
        cumulative_reward = 0
        episode_steps = 0

    episode_steps += 1

    if step % eval_step == 0:
        test_score = []
        for _ in range(num_test):
            eval_state = test_env.reset()
            eval_cumulative_reward = 0
            for _ in range(test_env.spec.max_episode_steps):
                eval_action = agent.get_deterministic_action(eval_state)
                eval_next_state, eval_reward, eval_done, _ = test_env.step(eval_action * test_env.action_space.high[0])
                eval_cumulative_reward += eval_reward
                eval_state = eval_next_state
                if eval_done:
                    break
            test_score.append(eval_cumulative_reward)
        writer.add_scalar("log/Test Reward", sum(test_score) / num_test, step)
        t2 = time()
        elapsed_time = t2 - t1
        elapsed_hour = int(elapsed_time // 3600)
        elapsed_min = int((elapsed_time % 3600) // 60)
        elapsed_sec = int((elapsed_time % 3600) % 60)
        print('Num Steps:{:<6}, Evaluate 10 average score {}, Time: {:02}:{:02}:{:02}'.format(
            step, sum(test_score) / num_test, elapsed_hour, elapsed_min, elapsed_sec
        ))


test_score = []
for _ in range(num_test):
    eval_state = test_env.reset()
    eval_cumulative_reward = 0
    for _ in range(test_env.spec.max_episode_steps):
        eval_action = agent.get_deterministic_action(eval_state)
        eval_next_state, eval_reward, eval_done, _ = test_env.step(eval_action * test_env.action_space.high[0])
        eval_cumulative_reward += eval_reward
        eval_state = eval_next_state
        if eval_done:
            break
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
