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

# init policy, valuefn
input_size = 4
output_size = 2
layer_size = 12
num_layers = 2
activation = nn.ReLU

# torch.set_default_dtype(torch.double)

env_name = "linear_z-v0"

seed = 0
policy = MLP(input_size, output_size, num_layers, layer_size, activation)

# model = PPOModelActHold(
#     policy=policy,
#     value_fn=MLP(input_size, 1, num_layers, layer_size, activation),
#     discrete=False,
#     hold_count = 10
# )

model = PPOModel(policy=policy, value_fn=MLP(input_size, 1, num_layers, layer_size, activation), discrete=False)

def reward_fn(s):
    if s[3] == 1:
        if s[0] > 2 and s[2] > 3:
            reward = 5.0
            s[3] = 0
        else:
            reward = -1.0

    elif s[3] == 0:
        if s[0] < -2 and s[2] < -3:
            reward = 5.0
            s[3] = 1
        else:
            reward = -1.0

    return reward, s

env_config = {
    "num_steps" : 500,
    "reward_fn" : reward_fn
}

arg_dict = {
    "env_name": env_name,
    "model": model,
    "act_var_schedule": [1],
    "seed": seed,  # int((time.time() % 1)*1e8),
    "total_steps" : 5e5,
    "epoch_batch_size": 2048,
    "gamma": 1,
    "pol_epochs": 10,
    "val_epochs": 10,
    "env_config": env_config
}

run_name = "debug2" + str(seed)

run_sg(arg_dict,
       ppo,
       run_name,
       "basic smoke test",
       "/data/seagul/"
       )

print("finished run ", run_name)
