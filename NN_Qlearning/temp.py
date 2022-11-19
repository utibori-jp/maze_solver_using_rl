import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Define a Neural Network
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        y = F.relu(self.l1(x))
        y = torch.sigmoid(self.l2(y))
        y = F.relu(self.l3(y))
        y = self.dropout(y)
        y = self.l4(y)
        return y

# set parameters
lr = 0.002
iters = 3000
torch.manual_seed(0)

# prepare dataset
batch_size, input_size, hidden_size, output_size = 10, 5, 10, 5
x = torch.randn(batch_size, input_size)
y = torch.sin(2 * np.pi * x) + torch.randn(batch_size, input_size)

# model
model = Net(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = lr)

# if you want to use "GPU" on mac, uncomment the following bolock.
# device = torch.device("mps")
# model.to(device)
# x = x.to(device)
# y = y.to(device)

# model training
model.train()
for i in range(iters):
    # compute prediction error
    y_pred = model.forward(x)
    loss = criterion(y, y_pred)

    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 300 == 0:
        print(loss)