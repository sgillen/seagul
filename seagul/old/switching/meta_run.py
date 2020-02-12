from multiprocessing import Process
import seagul.envs

# import time

import gym

env_name = "su_acro_drake-v0"

env = gym.make(env_name)

import torch
import torch.nn as nn


# init policy, valuefn
input_size = 4
output_size = 1
layer_size = 24
num_layers = 2
activation = nn.ReLU

from seagul.rl.run_utils import run_sg, run_and_save_bs
from seagul.rl.algos import ppo, ppo_switch
from seagul.rl.models import PPOModel, SwitchedPPOModel, PPOModelActHold
from seagul.nn import MLP, CategoricalMLP

proc_list = []

#torch.set_default_dtype(torch.double)
seed = 0

policy = MLP(input_size, output_size, num_layers, layer_size, activation)
    
model = PPOModelActHold(
    policy=policy,
    value_fn=MLP(input_size, 1, num_layers, layer_size, activation),
    discrete=False,
    hold_count = 1
)

#model = PPOModel(policy=policy, value_fn=MLP(input_size, 1, num_layers, layer_size, activation), discrete=False)


arg_dict = {
    "env_name": env_name,
    "model": model,
    "act_var_schedule": [.1],
    "seed": seed,  # int((time.time() % 1)*1e8),
    "total_steps" : 200*2048,
    "epoch_batch_size": 2048,
    "reward_stop" : 900,
    "gamma": 1,
    "pol_epochs": 10,
    "val_epochs": 10,
}


run_sg(arg_dict, ppo, None, 'lets see if we can learn to balance', "/data/data2/10_sat/")

    # p = Process(
    #     target=run_sg,
    #     args=(arg_dict, ppo, None, "ppo2 drake acrobot with an act hold of 20, to see if Nans go away..", "/data2/ppo2_test/"),
    # )
    # p.start()
    # proc_list.append(p)


# for p in proc_list:
#     print("joining")
#     p.join()
