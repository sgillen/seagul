from multiprocessing import Process
import seagul.envs

#import time

import gym


env_name = 'su_acro_drake-v0'

env = gym.make(env_name)


import torch
import torch.nn as nn


#init policy, valuefn
input_size = 4
output_size = 1
layer_size = 12
num_layers=2
activation=nn.Tanh

from seagul.rl.run_utils import run_sg, run_and_save_bs
from seagul.rl.algos import ppo, ppo_switch
from seagul.rl.models import PpoModel, switchedPpoModel, PpoModelActHold
from seagul.nn import MLP, CategoricalMLP
from seagul.sims.cartpole import LQRControl


torch.set_default_dtype(torch.double)
proc_list = []



for seed in range(6,10):

    policy = MLP(input_size, output_size, num_layers, layer_size, activation)

    model = PpoModelActHold(
        policy=policy,
        value_fn=MLP(input_size, 1, num_layers, layer_size, activation),
        discrete=False,
        hold_count = 200
    )

    # model = PpoModel(
    #     policy=policy,
    #     value_fn=MLP(input_size, 1, num_layers, layer_size, activation),
    #     discrete=False,
    # )

    arg_dict = {
        'env_name' : env_name,
        'model' : model,
        'action_var_schedule' : [1,1],
        'seed' : seed, #int((time.time() % 1)*1e8),
        'num_epochs' : 500,
        'gamma' : 1,
        'p_epochs' : 10,
        'v_epochs' : 10,
    }
    
    run_name = "seed" + str(seed)


    run_sg(arg_dict, ppo, run_name, 'baselines for drake', "/data/drake_acro2/")
    #    import ipdb; ipdb.set_trace()
    #    run_sg(arg_dict, ppo, run_name, 'running my own POO as a baseline', "/data/drake_acro1/")
    
#     p = Process(target=run_sg, args=(arg_dict, ppo, run_name, 'just a debug run', "/data/drake_acro1/"))
#     p.start()
#     proc_list.append(p)

    
# for p in proc_list:
#     print("joining")
#     p.join()
