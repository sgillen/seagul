import gym
import seagul.envs

from seagul.rl.run_utils import run_sg, run_and_save_bs
from seagul.rl.algos import sac
from seagul.rl.models import SACModel, PPOModelActHold
from seagul.nn import MLP, CategoricalMLP, DummyNet


import torch
import torch.nn as nn


import numpy as np
from numpy import pi

from multiprocessing import Process

## init policy, valuefn
input_size = 4
output_size = 2
layer_size = 12
num_layers = 2
activation = nn.ReLU

#torch.set_default_dtype(torch.double)
proc_list = []

env_name = "linear_z-v0"

for seed in [0,1,2,3]:

    
    policy = MLP(input_size, output_size * 2, num_layers, layer_size, activation)
    value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
    q1_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)
    q2_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)
    model = SACModel(policy, value_fn, q1_fn, q2_fn, 1)



    arg_dict = {
        "env_name": env_name,
        "model": model,
        "seed": seed,  # int((time.time() % 1)*1e8),
        "total_steps" : 5e4,
    }

    run_name = "linear1" + str(seed)

    p = Process(
        target=run_sg,
        args=(
            arg_dict,
            sac,
            run_name,
            "basic smoke test",
            "/data/seagul_sac/",
        ),
    )
    p.start()
    proc_list.append(p)

for p in proc_list:
    print("joining")
    p.join()


print("finished run ", run_name)
