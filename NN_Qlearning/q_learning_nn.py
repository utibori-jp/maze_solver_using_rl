import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from common.gridworld import GridWorld

# Define a Neural Network
class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        y = F.relu(self.l1(x))
        y = torch.sigmoid(self.l2(y))
        y = self.l3(y)
        return y

class QLearningAgent:
    def __init__(self, input_size, hidden_size, output_size):
        self.gamma = 0.9
        self.lr = 0.02
        self.epsilon = 0.1
        self.action_size = 4

        self.qnet = QNet(input_size, hidden_size, output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.qnet.parameters(), lr = lr)

    def get_action(self, state_vec): # ε-greedy
        if np.random.rand() < self.epsilon: 
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet(state_vec)
            return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q = torch.zeros(1)
        else:
            next_qs = self.qnet(next_state)
            next_q = next_qs.max(axis=1).values
            next_q = next_q.detach()

        target = self.gamma * next_q + reward
        qs = self.qnet(state)
        q = qs[:, action]
        loss = self.criterion(target, q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.data

def one_hot(state):
    HEIGHT, WIDTH = 4, 5
    vec = np.zeros(HEIGHT * WIDTH, dtype = np.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :]

def to_tensor(state):
    state = one_hot(state)
    state = torch.from_numpy(state).clone()
    return state

def torch_fix_seed(seed = 3407):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# set params
batch_size, input_size, hidden_size, output_size = 1, 20, 100, 4
lr = 0.002
torch_fix_seed()

# q_learning and
env = GridWorld()
agent = QLearningAgent(input_size, hidden_size, output_size)

episodes = 1000
loss_history = []
for episode in range(episodes):
    state = env.reset()
    tensor_state = to_tensor(state)
    total_loss, cnt = 0, 0
    done = False

    while not done:
        action = agent.get_action(tensor_state)
        if isinstance(action, int):
            action = torch.tensor(action, dtype = torch.int8)
        next_state, reward, done = env.step(action)
        next_tensor_state = to_tensor(next_state)
        loss = agent.update(tensor_state, action, reward, next_tensor_state, done)
        total_loss += loss
        cnt += 1
        state = next_state

    average_loss = total_loss / cnt
    loss_history.append(average_loss)

# plot loss transition
plt.xlabel("episode")
plt.ylabel("loss")
plt.plot(range(len(loss_history)), loss_history)
plt.show()

Q = {}
for state in env.states():
    tensor_state = to_tensor(state)
    for action in env.action_space:
        q = agent.qnet(tensor_state)[:, action]
        Q[state, action] = float(q.data)
env.render_q(Q)
