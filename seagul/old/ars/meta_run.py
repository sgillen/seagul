from multiprocessing import Process
import seagul.envs

# import time

import gym

env_name = "Walker2d-v2"
env = gym.make(env_name)

import torch
import torch.nn as nn

# init policy, valuefn
input_size = 17
output_size = 6
layer_size = 64
num_layers = 3
activation = nn.ReLU

from seagul.rl.run_utils import run_sg, run_and_save_bs
from seagul.rl.algos import ppo, ppo_switch
from seagul.rl.models import PPOModel, SwitchedPPOModel, PPOModelActHold
from seagul.nn import MLP, CategoricalMLP

torch.set_default_dtype(torch.double)
proc_list = []

for seed in [0]:

    policy = MLP(input_size, output_size, num_layers, layer_size, activation)

    # model = PPOModelActHold(
    #     policy=policy,
    #     value_fn=MLP(input_size, 1, num_layers, layer_size, activation),
    #     discrete=False,
    #     hold_count = 200
    # )

    model = PPOModel(policy=policy, value_fn=MLP(input_size, 1, num_layers, layer_size, activation), discrete=False)

    arg_dict = {
        "env_name": env_name,
        "model": model,
        "action_var_schedule": [1, 1],
        "seed": seed,  # int((time.time() % 1)*1e8),
        "num_epochs": 1000,
        "epoch_batch_size": 2048,
        "gamma": 1,
        "p_epochs": 10,
        "v_epochs": 10,
    }

    run_name = "ppo" + str(seed)

    #    run_sg(arg_dict, ppo, run_name, 'run with 100 epochs, torque limit', "/data/drake_acro_final/")

    p = Process(
        target=run_sg,
        args=(arg_dict, ppo, run_name, "ppo for walker with nn policy, state norm turned on", "./data/walker/"),
    )
    p.start()
    proc_list.append(p)


for p in proc_list:
    print("joining")
    p.join()
