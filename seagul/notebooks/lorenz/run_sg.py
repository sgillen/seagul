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

## init policy, valuefn
input_size = 3
output_size = 3
layer_size = 12
num_layers = 2
activation = nn.ReLU

#torch.set_default_dtype(torch.double)
proc_list = []

env_name = "lorenz-v0"

for seed in [0,1,2,3]:
    policy = MLP(input_size, output_size, num_layers, layer_size, activation)
    
    model = PPOModelActHold(
        policy=policy,
        value_fn=MLP(input_size, 1, num_layers, layer_size, activation),
        discrete=False,
        hold_count = 10
    )

    #model = PPOModel(policy=policy, value_fn=MLP(input_size, 1, num_layers, layer_size, activation), discrete=False)


    arg_dict = {
        "env_name": env_name,
        "model": model,
        "act_var_schedule": [5],
        "seed": seed,  # int((time.time() % 1)*1e8),
        "total_steps" : 100*2048,
        "epoch_batch_size": 2048,
        "reward_stop" : 900,
        "gamma": 1,
        "pol_epochs": 10,
        "val_epochs": 10,
    }


    p = Process(
        target=run_sg,
        args=(
            arg_dict,
            ppo,
            run_name,
            "basic smoke test",
            "/data/full_act/",
        ),
    )
    p.start()
    proc_list.append(p)

for p in proc_list:
    print("joining")
    p.join()


print("finished run ", run_name)
