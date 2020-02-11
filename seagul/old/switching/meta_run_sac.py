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
from seagul.rl.algos import sac
from seagul.rl.models import SACModel, SACModelActHold
from seagul.nn import MLP, CategoricalMLP

proc_list = []

#torch.set_default_dtype(torch.double)
seed = 0

policy = MLP(input_size, output_size*2, num_layers, layer_size, activation)
value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
q1_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)
q2_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)

# model = SACModelActHold(
#     policy=policy,
#     value_fn = value_fn,
#     q1_fn = q1_fn,
#     q2_fn = q2_fn,
#     act_limit = 5,
#     hold_count = 200,
# )

model = SACModel(
    policy=policy,
    value_fn = value_fn,
    q1_fn = q1_fn,
    q2_fn = q2_fn,
    act_limit = 5,
)


arg_dict = {
    "env_name": env_name,
    "model": model,
    "seed": seed,  # int((time.time() % 1)*1e8),
    "total_steps" : 5e5,
    "exploration_steps" : 10000,
    "min_steps_per_update" : 200,
    "reward_stop" : 1500,
    "gamma": 1,
}


run_sg(arg_dict, sac, None, 'back to 200 ah', "/data/data2/sac/")

    # p = Process(
    #     target=run_sg,
    #     args=(arg_dict, ppo, None, "ppo2 drake acrobot with an act hold of 20, to see if Nans go away..", "/data2/ppo2_test/"),
    # )
    # p.start()
    # proc_list.append(p)


# for p in proc_list:
#     print("joining")
#     p.join()
