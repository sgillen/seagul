import gym
import seagul.envs

env_name = "su_acro_drake-v0"
env = gym.make(env_name)

from seagul.rl.run_utils import run_sg, run_and_save_bs
from seagul.rl.algos import ppo, ppo_switch
from seagul.rl.models import PPOModel, SwitchedPPOModel, SwitchedPPOModelActHold
from seagul.nn import MLP, CategoricalMLP, DummyNet

import torch
import torch.nn as nn

import numpy as np
from numpy import pi

from multiprocessing import Process

# init policy, value fn
input_size = 4
output_size = 1
layer_size = 12
num_layers = 2
activation = nn.ReLU
proc_list = []

for seed in [0,1,2,3]:
    env_name = "su_acro_drake-v0"
    env = gym.make(env_name)

    torch.set_num_threads(1)

    def control(q):
        k = np.array([[1316.85000612, 555.41763935, 570.32667002, 272.57631536]], dtype=np.float32)
        # k = np.array([[278.44223126, 112.29125985, 119.72457377,  56.82824017]])
        gs = np.array([pi, 0, 0, 0], dtype=np.float32)

        return (-k.dot(gs - np.asarray(q)))

    model = SwitchedPPOModelActHold(
        # policy = MLP(input_size, output_size, num_layers, layer_size, activation),
        policy=torch.load("warm/ppo2_warm_pol"),
        value_fn=torch.load("warm/ppo2_warm_val"),
        # MLP(input_size, 1, num_layers, layer_size, activation),
        gate_fn=torch.load("warm/gate_fn_ppo2_nz128"),
        nominal_policy=control,
        hold_count=20,
    )

    arg_dict = {
        "env_name": env_name,
        "model": model,
        "total_steps": 500*2048,
        "epoch_batch_size": 2048,
        "act_var_schedule": [2, 2],
        "gate_var_schedule": [0.1, 0.1],
        "gamma": 1,
        "seed": seed,
        "reward_stop" : 1500,
    }

    run_name = "25_ppo2" + str(seed)

    #  import ipdb; ipdb.set_trace()
    # run_sg(arg_dict, ppo_switch, run_name, 'reasonable torque limits, and a new but cheaty warm start', "/data/switch4/")

    run_sg(
        arg_dict,
        ppo_switch,
        run_name,
        "trying to replicate earlier results that use ppo with ppo2",
        "/data/drake_ppo22/",
    )

    print("finished run ", run_name)
