import gym
import seagul.envs

from seagul.rl.run_utils import run_sg, run_and_save_bs
from seagul.rl.algos import ppo, ppo_switch
from seagul.rl.models import PPOModel, PPOModelActHold
from seagul.nn import MLP, CategoricalMLP, DummyNet


import torch
import torch.nn as nn


import numpy as np
from numpy import pi

from multiprocessing import Process

## init policy, valuefn

#torch.set_default_dtype(torch.double)
proc_list = []

#env_name = "linear_z-v0"
env_name = "lorenz-v0"
env = gym.make(env_name)

input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]
layer_size = 12
num_layers = 2
activation = nn.ReLU



for seed in [0,1,2,3]:
    policy = MLP(input_size, output_size, num_layers, layer_size, activation)
    
    # model = PPOModelActHold(
    #     policy=policy,
    #     value_fn=MLP(input_size, 1, num_layers, layer_size, activation),
    #     discrete=False,
    #     hold_count = 10
    # )

    model = PPOModel(policy=policy, value_fn=MLP(input_size, 1, num_layers, layer_size, activation), discrete=False)


    arg_dict = {
        "env_name": env_name,
        "model": model,
        "act_var_schedule": [5],
        "seed": seed,  # int((time.time() % 1)*1e8),
        "total_steps" : 1e6,
        "epoch_batch_size": 2048,
        "gamma": 1,
        "pol_epochs": 10,
        "val_epochs": 10,
    }

    run_name = "lorenz_again" + str(seed)

    p = Process(
        target=run_sg,
        args=(
            arg_dict,
            ppo,
            run_name,
            "basic smoke test",
            "/data2/seagul/",
        ),
    )
    p.start()
    proc_list.append(p)

for p in proc_list:
    print("joining")
    p.join()


print("finished run ", run_name)
