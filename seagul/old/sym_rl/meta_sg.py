from multiprocessing import Process
import seagul.envs

# import time
import gym

# Do this first because otherwise drake can break
env_name = "Pendulum-v0"
env = gym.make(env_name)

import torch
import torch.nn as nn

from seagul.rl.run_utils import run_sg
from seagul.rl.algos import ppo, ppo_sym
from seagul.rl.models import PPOModel, PPOModelActHold
from seagul.nn import MLP, CategoricalMLP

input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]
layer_size = 32
num_layers = 2
activation = nn.ReLU

torch.set_default_dtype(torch.double)
proc_list = []

for seed in [4, 5, 6, 7]:

    policy = MLP(input_size, output_size, num_layers, layer_size, activation)

    # model = PPOModelActHold(
    #     policy=policy,
    #     value_fn=MLP(input_size, 1, num_layers, layer_size, activation),
    #     discrete=False,
    #     hold_count = 200
    # )

    model = PPOModel(
        policy=policy, value_fn=MLP(input_size, output_size, num_layers, layer_size, activation), discrete=False
    )

    arg_dict = {
        "env_name": env_name,
        "model": model,
        "action_var_schedule": [0.7],
        "seed": seed,  # int((time.time() % 1)*1e8),
        "num_epochs": 800,
        "env_timesteps": 200,
        "epoch_batch_size": 1024,
        "gamma": 0.95,
        # "v_epochs": 10,
        # "policy_batch_size": 2048,
        # "value_batch_size": 2048,
    }

    run_name = "sym_mean" + str(seed)

    #    run_sg(arg_dict, ppo, run_name, '', "/data/sym_comp/ppo")

    p = Process(target=run_sg, args=(arg_dict, ppo_sym, run_name, "update means after the fact", "/data/sym_comp/"))
    p.start()
    proc_list.append(p)


for p in proc_list:
    print("joining")
    p.join()


# for seed in [4,5,6,7]:

#     policy = MLP(input_size, output_size, num_layers, layer_size, activation)

#     # model = PPOModelActHold(
#     #     policy=policy,
#     #     value_fn=MLP(input_size, 1, num_layers, layer_size, activation),
#     #     discrete=False,
#     #     hold_count = 200
3
#     model = PPOModel(
#         policy=policy,
#         value_fn=MLP(input_size, output_size, num_layers, layer_size, activation),
#         discrete=False,
#     )

#     arg_dict = {
#         'env_name' : env_name,
#         'model' : model,
#         'action_var_schedule' : [.5],
#         'seed' : seed, #int((time.time() % 1)*1e8),
#         'num_epochs' : 800,
#         'epoch_batch_size': 1024,
#         'env_timesteps' : 200,
#         'gamma' : .95,
#         'p_epochs' : 32,
#         'v_epochs' : 32,
#         'policy_batch_size' : 512,
#         'value_batch_size' : 512
#     }

#     run_name = "ppo" + str(seed)


# #    run_sg(arg_dict, ppo, run_name, '', "/data/sym_comp/ppo")

#     p = Process(target=run_sg, args=(arg_dict, ppo, run_name, '', "/data/sym_comp/"))
#     p.start()
#     proc_list.append(p)


# for p in proc_list:
#    print("joining")
#    p.join()
