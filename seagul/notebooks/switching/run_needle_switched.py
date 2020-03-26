from multiprocessing import Process
from seagul.rl.run_utils import run_sg
from seagul.rl.sac.sacn_adaptive import sac_switched
import torch
import torch.nn as nn
from seagul.nn import MLP
from seagul.rl.models import SACModelSwitch
import numpy as np
from scipy.stats import multivariate_normal
from seagul.integration import rk4

trial_name = input('Trial name please:\n')

input_size = 4
output_size = 1
layer_size = 32
num_layers = 4
activation = nn.ReLU

m1 = 1; m2 = 1
l1 = 1; l2 = 1
lc1 = .5; lc2 = .5
I1 = .2; I2 = 1.0
g = 9.8
max_torque = 5
lqr_max_torque = 5
max_t = 10.0


def control(q):
    q = torch.as_tensor(q, dtype=torch.float32)
    k = torch.tensor([[-1649.86567367, -460.15780461, -716.07110032, -278.15312267]])
    gs = torch.tensor([np.pi / 2, 0, 0, 0])
    return (-k * (q - gs)).sum(dim=1).detach()

def reward_fn_sin(s, a):
    reward = (np.sin(s[0]) + np.sin(s[0] + s[1]))
    return reward, False

def reward_fn_gauss(s, a):
    return multivariate_normal.pdf(s, mean=[np.pi / 2, 0, 0, 0], cov=[1, 1, 1, 1]), False


policy = MLP(input_size, output_size * 2, num_layers, layer_size, activation)
value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
q1_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)
q2_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)
model = SACModelSwitch(policy, value_fn, q1_fn, q2_fn, 25, balance_controller=control
                       , hold_count=20, gate_fn=torch.load("warm/gate25_rk"))

env_config = {
    "init_state": [-np.pi / 2, 0, 0, 0],
    "max_torque": max_torque,
    "init_state_weights": [np.pi, np.pi, 0, 0],
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
    "integrator" : rk4
}

proc_list = []
for seed in np.random.randint(0, 2 ** 32, 8):
    alg_config = {
        "env_name": "su_acrobot-v0",
        "total_steps": 2e6,
        "model": model,
        "seed": seed,
        "goal_state": np.array([np.pi / 2, 0, 0, 0]),
        "goal_lookback": 10,
        "goal_thresh": .25,
        "alpha": .05,
        "needle_lookup_prob": .8,
        "exploration_steps": 50000,
        "gate_update_freq": float('inf'),
        "gate_x": torch.as_tensor(torch.load('warm/X25_rk')),
        "gate_y": torch.as_tensor(torch.load('warm/Y25_rk')),
        "env_config": env_config,
        "min_steps_per_update" : 500,
        "sgd_batch_size": 128,
        "replay_batch_size" : 4096,
        "iters_per_update": 4,
        "use_gpu": False,
        "gamma" : .99,
        "replay_buf_size" : int(500000),
    }

    # sac_switched(**alg_config)

    p = Process(
        target=run_sg,
        args=(alg_config, sac_switched, "trial" + str(seed), "", "/data_needle/" + trial_name + "/"),
    )
    p.start()
    proc_list.append(p)

for p in proc_list:
    print("joining")
    p.join()
