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

    env_name = 'su_cartpole-v0'
    env = gym.make(env_name)
    
    
    model = switchedPpoModel(
        policy = MLP(input_size, output_size, num_layers, layer_size, activation),
        value_fn = MLP(input_size, 1, num_layers, layer_size, activation),
        gate_fn  = torch.load("gate_fn_v"),
        nominal_policy=LQRControl,
        env=env
    )
        
    arg_dict = {
        'env_name' : env_name,
        'model' : model,
        'num_epochs' : 500,
        'action_var_schedule' : [10,0],
        'gate_var_schedule'   : [1,0],
        'seed': seed
    }


    run_name = "ppo_rew_d" + str(seed)
    p = Process(target=run_sg, args=(arg_dict, ppo, run_name, 'new warm start net', "/data/cartpole2/"))
    p.start()
    proc_list.append(p)

for p in proc_list:
    print("joining")
    p.join()
