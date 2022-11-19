import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen = buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.input_size = 20
        self.hidden_size = 100
        self.output_size = 4
        self.l1 = nn.Linear(self.input_size, self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.l3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        y = F.relu(self.l1(x))
        y = torch.sigmoid(self.l2(y))
        y = self.l3(y)
        return y

class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.001
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 2

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet()
        self.qnet_target = QNet()
        self.optimizer = optim.Adam(self.qnet.parameters(), self.lr)

    def get_action(self, state):
        # ε-greedy
        if np.random.rand() < self.epsilon: 
            return np.random.choice(self.action_size)
        else:
            state = state[np.newaxis, :]
            qs = self.qnet(state)
            return qs.data.argmax()

agent = DQNAgent()
state = (2, 0)
print(agent.get_action(state))
