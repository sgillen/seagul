from multiprocessing import Process
from seagul.rl.run_utils import run_sg
import seagul.envs
from seagul.rl.algos.sac_needle import sac
import torch
import torch.nn as nn
from seagul.nn import MLP
from seagul.rl.models import SACModel
import numpy as np
from scipy.stats import multivariate_normal 

trial_name = input('Trial name please:\n')

input_size = 6
output_size = 1
layer_size = 32
num_layers = 2
activation = nn.ReLU

policy = MLP(input_size, output_size * 2, num_layers, layer_size, activation)
value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
q1_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)
q2_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)
model = SACModel(policy, value_fn, q1_fn, q2_fn, 5)

m1 = 1; m2 = 1
l1 = 1; l2 = 1
lc1 = .5; lc2 = .5
I1 = .2; I2 = 1.0
g = 9.8
max_torque = 5
max_t = 10.0

def control(q):
    k = np.array([[-1649.86567367, -460.15780461, -716.07110032, -278.15312267]])
    gs = np.array([np.pi / 2, 0, 0, 0])
    return -k.dot(q - gs)

def reward_fn_sin(s, a):
    reward = (np.sin(s[0]) + np.sin(s[0] + s[1]))
    return reward, False

def reward_fn_gauss(s,a):
    return multivariate_normal.pdf(s, mean=[pi/2,0,0,0], cov=[1,1,1,1])

    

env_config = {
    "init_state": [-np.pi / 2, 0, 0, 0],
    "max_torque": max_torque,
    "init_state_weights": [2, 2, 0, 0],
    "dt": .01,
    "reward_fn": reward_fn_sin,
    "max_t": max_t,
    "m2": m2,
    "m1": m1,
    "l1": l1,
    "lc1": lc1,
    "lc2": lc2,
    "i1": I1,
    "i2": I2,
    "act_hold": 20,
    "gate_fn": torch.load("../switching/warm/lqr_gate_better"),
    "controller": control
}

proc_list = []
for seed in np.random.randint(0, 2**32,1):

    alg_config = {
        "env_name" : "su_acroswitchsin-v0",
        "total_steps" : 500000,
        "model" : model,
        "seed" : seed,
        "goal_state" : np.array([0,1,1,0,0,0]),
        "goal_lookback" : 10,
        "goal_thresh" : 1.5,
        "iters_per_update" : float('inf'),
        "exploration_steps" : 50000,
        "env_config" : env_config
    }

run_sg(alg_config, sac, "smoke_test"+str(seed), "", "/data_needle/" + trial_name)


