import random
import gymnasium as gym
import numpy as np
from tqdm import trange


def init_q_table(obs_space, action_space):
    return np.zeros((obs_space, action_space))


def epsilon_greedy(q_table, state, epsilon):
    random_int = random.uniform(0, 1)
    if random_int > epsilon:
        action = np.argmax(q_table[state])
    else:
        action = env_train.action_space.sample()

    return action


def greedy(q_table, state):
    return np.argmax(q_table[state])


def train_agent(env, episodes, max_steps, q_table, min_epsilon, max_epsilon, decay_epsilon):
    for episode in trange(episodes):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_epsilon * episode)
        state = env.reset()[0]

        for step in range(max_steps):
            action = epsilon_greedy(q_table, state, epsilon)
            new_state, reward, terminated, truncated, info = env.step(action)
            q_table[state][action] = q_table[state][action] + learning_rate * (reward + gamma * np.max(q_table[new_state]) - q_table[state][action])

            if terminated or truncated:
                break

            state = new_state

    env.close()

    return q_table


def evaluate_agent(env, max_steps, episodes, q_table, seed):
    episode_rewards = []

    for episode in trange(episodes):
        if seed:
            state = env.reset(seed=seed[episode])[0]
        else:
            state = env.reset()[0]

        total_rewards_ep = 0

        for step in range(max_steps):
            action = greedy(q_table, state)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break

            state = new_state

        episode_rewards.append(total_rewards_ep)

    env.close()
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


env_train = gym.make('Taxi-v3')
env_test = gym.make('Taxi-v3', render_mode='human')
training_episodes = 10000
max_steps = 200
learning_rate = 0.7
gamma = 0.95
max_epsilon = 1
min_epsilon = 0.05
decay_epsilon = 0.0005

eval_episodes = 10
eval_seeds = []

q_table = init_q_table(env_train.observation_space.n, env_train.action_space.n)
q_table_trained = train_agent(env_train, training_episodes, max_steps, q_table, min_epsilon, max_epsilon, decay_epsilon)

mean_reward, std_reward = evaluate_agent(env_test, max_steps, eval_episodes, q_table_trained, eval_seeds)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")