import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random


env = gym.make('CliffWalking-v0')
state_space = env.observation_space.n
action_space = env.action_space.n

# Модель DQN
class DQN(nn.Module):
    def __init__(self, state_space, action_space):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_space, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_space)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Функція навчання
def train_dqn(env, model, episodes, gamma, epsilon, epsilon_decay, optimizer, batch_size):
    replay_buffer = deque(maxlen=10000)
    rewards = []

    for episode in range(episodes):
        state = env.reset()[0]
        state = np.eye(state_space)[state]
        total_reward = 0
        done = False

        while not done:
            # Вибір дії
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = model(torch.tensor(state, dtype=torch.float32))
                action = torch.argmax(q_values).item()

            # Виконання дії
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.eye(state_space)[next_state]  # One-hot encoding
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Навчання з буферу
            if len(replay_buffer) > batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards_batch, next_states, dones = zip(*batch)

                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.long)
                rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                q_values = model(states)
                next_q_values = model(next_states).detach()
                target_q = rewards_batch + gamma * torch.max(next_q_values, dim=1)[0] * (1 - dones)

                loss = nn.functional.mse_loss(q_values.gather(1, actions.unsqueeze(1)).squeeze(), target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        rewards.append(total_reward)
        epsilon *= epsilon_decay
    return rewards


def plot_rewards(rewards):
    plt.plot(rewards)
    plt.title('Середня винагорода')
    plt.xlabel('Епізоди')
    plt.ylabel('Винагорода')
    plt.show()


model = DQN(state_space, action_space)
optimizer = optim.Adam(model.parameters(), lr=0.001)
rewards = train_dqn(env, model, 500, 0.99, 1.0, 0.995, optimizer, 32)
plot_rewards(rewards)
