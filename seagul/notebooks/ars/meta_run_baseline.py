import sys

# sys.path.append('/home/sgillen/Downloads/drake/underactuated/src/')
# sys.path.append('/opt/drake/lib/python3.6/site-packages/')


import seagul.envs
import gym


# We do this here so that drake envs don't break https://github.com/RobotLocomotion/drake/issues/12073

env_name = "Hopper-v2"
env = gym.make(env_name)

from seagul.rl.run_utils import run_and_save_bs

# from seagul.rl.algos import ppo, ppo_switch
# from seagul.rl.models import PPOModel, SwitchedPPOModel
# from seagul.nn import MLP, CategoricalMLP


from multiprocessing import Process
import seagul.envs

import time

import torch
import torch.nn as nn


input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]
layer_size = 64
num_layers = 2
activation = nn.ReLU

torch.set_default_dtype(torch.double)
proc_list = []

for seed in range(1):

    # seed = int((time.time() % 1)*1e8)

    arg_dict = {
        "env": "Walker2d-v2",
        "alg": "ppo2",
        "network": "mlp",
        "num_timesteps": "8e6",
        "num_env": "8",
        "num_layers": "2",
        "num_hidden": "64",
        "seed": str(seed),
    }

    run_name = "bs_ppo_v2_pythran" + str(seed)
    run_and_save_bs(arg_dict, run_name, "baselines ppo with nn", "/data/walker/")
    #    run_and_save_bs(arg_dict, run_name, 'baseline for ppo2', "/data/drake_base/")
    # p = Process(target=run_and_save_bs, args=(arg_dict, run_name, 'baseline for ppo2 with act_hold = 200', "/data/drake_base/"))
    # p.start()
#    p.join()

# for seed in range(6,10):

#     #seed = int((time.time() % 1)*1e8)

#     arg_dict = {
#         'env': 'su_acrobot-v0',
#         'alg': 'ppo2',
#         'network': 'mlp'
#         'num_timesteps': '3e6',
#         'num_env': '4',
#         'num_layers': '2',
#         'num_hidden': '24',
#         'seed' :  str(seed)
#     }


#     run_name = "baseline_ppo2_nh" + str(seed)
#     p = Process(target=run_and_save_bs, args=(arg_dict, run_name, 'baseline ppo with no action hold', "/data/acrobot_switched4/"))
#     p.start()
#     p.join()
