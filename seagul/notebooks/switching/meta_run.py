import seagul.envs
import torch.nn as nn
import numpy as np
from numpy import pi
from multiprocessing import Process
from seagul.rl.run_utils import run_sg
from seagul.rl.ppo import ppo, PPOModel
from seagul.nn import MLP
import torch

# init policy, valuefn
input_size = 4
output_size = 1
layer_size = 32
num_layers = 1
activation = nn.ReLU

proc_list = []
trial_num = input("What trial is this??\n")

m1 = 1; m2 = 1
l1 = 1; l2 = 1
lc1 = .5; lc2 = .5
I1 = .2; I2 = 1.0
g = 9.8

# m1 = 1; m2 = 1
# l1 = 1; l2 = 2
# lc1 = .5; lc2 = 1
# I1 = .083; I2 = .33
# g = 9.8

def control(q):
    k = np.array([[-1649.86567367, -460.15780461, -716.07110032, -278.15312267]])
    gs = np.array([pi / 2, 0, 0, 0])
    return -k.dot(q - gs)

def reward_fn(s, a):
    reward = 1e-2*(np.sin(s[0]) + np.sin(s[0] + s[1]))
    return reward, False

def reward_fn_sin(s,a):
    reward = (np.sin(s[0]) + np.sin(s[0] + s[1]))
    return reward, False

max_torque = 25
max_t = 10
act_var = 3.0

env_name = "su_acroswitch-v0"

for seed in np.random.randint(0, 2**32, 4):

            model = PPOModel(
                policy=MLP(input_size, output_size, layer_size, num_layers),
                value_fn=MLP(input_size, output_size, layer_size, num_layers),
            )

            env_config = {
                "init_state": [-pi / 2, 0, 0, 0],
                "max_torque": max_torque,
                "init_state_weights": [0, 0, 0, 0],
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
                "gate_fn": torch.load("warm/lqr_gate_better"),
                "controller": control
            }

            alg_config = {
                "env_name": env_name,
                "model": model,
                "act_var_schedule": [act_var],
                "seed": int(seed),  # int((time.time() % 1)*1e8),
                "total_steps" : 5e5,
                "reward_stop" : None,
                "gamma": 1,
                "pol_epochs": 30,
                "val_epochs": 30,
                "env_config" : env_config,
                }

            run_name = "swingup" + str(seed)

            #run_sg(alg_config, ppo, run_name , "normal reward", "/data4/switching_sortof/trial" + str(trial_num) + "_t" +  str(act_var) + "/")
            p = Process(target=run_sg, args=(alg_config, ppo, run_name , "warm_start", "/data5/acro2/trial" + str(trial_num) + "_v" + str(act_var) + "t" + str(max_t) + "/"))
            p.start()
            proc_list.append(p)


for p in proc_list:
    p.join()
