if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from common.gridworld import GridWorld
from collections import defaultdict
import policy_eval
from policy_iter import policy_iter, greedy_policy
from value_iter import value_iter

env = GridWorld()
algorithm = "value_iter"
gamma = 0.9

if algorithm == "policy_eval":
    pi = defaultdict(lambda: { 0:0.25, 1: 0.25, 2:0.25, 3: 0.25})
    V = defaultdict(lambda: 0)
    V = policy_eval(pi, V, env, gamma)
    env.render_v(V, pi)
elif algorithm == "policy_iter":
    pi = policy_iter(env, gamma)
elif algorithm == "value_iter":
    V = defaultdict(lambda: 0)
    V = value_iter(V, env, gamma)
    pi = greedy_policy(V, env, gamma)
    env.render_v(V, pi)
    