from seagul.rl.run_utils import run_sg, run_and_save_bs
from seagul.rl.algos import ppo, ppo_switch
from seagul.rl.models import PpoModel, switchedPpoModel
from seagul.nn import MLP, CategoricalMLP
from seagul.sims.cartpole import LQRControl
from multiprocessing import Process
import seagul.envs


import torch
import torch.nn as nn

import gym

## init policy, valuefn
input_size = 4
output_size = 1
layer_size = 24
num_layers=3
activation=nn.ReLU

torch.set_default_dtype(torch.double)
proc_list = []

for seed in range(4):

    arg_dict = {
        'env': 'su_cartpole-v0',
        'alg': 'ppo2',
        'network': 'mlp',
        'num_timesteps': '1e6',
        'num_env': '1',
        'num_layers': '3',
        'num_hidden': '12',
        'seed' : str(seed)
    }


    run_name = "seed" + str(seed)    
    p = Process(target=run_and_save_bs, args=(arg_dict, run_name, 'what is life?', "/data/cartpole_baseline4/"))
    p.start()
    proc_list.append(p)


