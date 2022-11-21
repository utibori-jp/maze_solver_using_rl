import numpy as np
import torch


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