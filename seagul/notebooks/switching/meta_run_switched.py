import gym
import seagul.envs


from seagul.rl.run_utils import run_sg, run_and_save_bs
from seagul.rl.algos import ppo, ppo_switch
from seagul.rl.models import PPOModel, SwitchedPPOModel, SwitchedPPOModelActHold
from seagul.nn import MLP, CategoricalMLP, DummyNet

import torch
import torch.nn as nn

import numpy as np
from numpy import pi

from multiprocessing import Process
import time

trial_num = input("what trial is this?")

# init policy, value fn
input_size = 4
output_size = 1
layer_size = 32
num_layers = 2
activation = nn.ReLU
proc_list = []

m1 = 1
m2 = 1
l1 = 1
l2 = 2
lc1 = .5
lc2 = 1
I1 = .083
I2 = .33
g = 9.8

k = np.array([[-787.27034288, -321.29901324, -342.50908801, -158.94341213]])

def control(q):
    gs = np.array([pi / 2, 0, 0, 0])
    return -k.dot(q - gs)

def reward_fn(s, a):
    reward = np.sin(s[0]) + 2 * np.sin(s[0] + s[1])
    return reward, False

max_torque = 25

env_config = {"init_state": [0, 0, 0, 0],
              "max_torque": max_torque,
              "init_state_weights": [0, 0, 0, 0],
              "max_t": 2.5,
              "m2": m2,
              "m1": m1,
              "l1": l1,
              "lc1": lc1,
              "lc2": lc2,
              "i1": I1,
              "i2": I2,
              "reward_fn": reward_fn,
              "act_hold": 1
              }


for seed in np.random.randint(0, 2**32,4):

    env_name = "su_acrobot-v0"
    env = gym.make(env_name, **env_config)

    run_name = "newnew"



    model = SwitchedPPOModelActHold(
        # policy = MLP(input_size, output_size, num_layers, layer_size, activation),
        policy=MLP(input_size, output_size, layer_size, num_layers),
        value_fn=MLP(input_size, output_size, layer_size, num_layers),
        gate_fn=torch.load("warm/gate_fn_good"),
        nominal_policy=control,
        hold_count=0,
    )


    alg_config = {
        "env_name": env_name,
        "model": model,
        "total_steps": 5e4,
        "epoch_batch_size": 2048,
        "act_var_schedule": [3],
        "gate_var_schedule": [0.1, 0.1],
        "gamma": 1,
        "seed": seed,
        "reward_stop": 1500,
        "pol_epochs": 300,
        "val_epochs": 10,
        "env_config": env_config
    }

    run_sg(alg_config, ppo_switch, run_name, "debug" , "/data/debug")

#     p = Process(
#         target=run_sg,
#         args=(
#             alg_config,
#             ppo_switch,
#             run_name,
#             "testing new gate",
#             "/data4/trial" + trial_num + "/",
#         ),
#     )
#     p.start()
#     proc_list.append(p)
#
# for p in proc_list:
#     print("joining")
#     p.join()

print("finished run ", run_name)
