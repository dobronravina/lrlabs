import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict

# 1. Створення середовища Frozen Lake
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv

env = FrozenLakeEnv(is_slippery=True)
n_states = env.observation_space.n
n_actions = env.action_space.n

# 2. Функція для ітераційного оцінювання стратегії
def policy_evaluation(policy, env, gamma=0.9, theta=1e-6):
    V = np.zeros(n_states)
    while True:
        delta = 0
        for s in range(n_states):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V

# 3. Рівноймовірна стратегія (equiprobable policy)
def create_uniform_policy():
    policy = np.ones((n_states, n_actions)) / n_actions
    return policy

# 4. Функція для візуалізації теплової карти
def plot_value_function(V, title="Value Function"):
    grid_size = int(np.sqrt(len(V)))
    value_matrix = V.reshape((grid_size, grid_size))
    plt.figure(figsize=(6, 6))
    plt.imshow(value_matrix, cmap="coolwarm", interpolation="none")
    plt.colorbar()
    plt.title(title)
    plt.show()

# Виконання оцінювання стратегії
uniform_policy = create_uniform_policy()
V_uniform = policy_evaluation(uniform_policy, env)

# Візуалізація отриманих значень
plot_value_function(V_uniform, title="Value Function for Equiprobable Policy")
