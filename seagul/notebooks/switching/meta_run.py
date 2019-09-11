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
activation=nn.Tanh

torch.set_default_dtype(torch.double)
proc_list = []

for seed in range(4):
        

    model = PpoModel(
        policy=MLP(input_size, output_size, num_layers, layer_size, activation),
        value_fn=MLP(input_size, 1, num_layers, layer_size, activation),
        discrete=False
    )

    arg_dict = {
        'env_name' : 'su_cartpole-v0',
        'model' : model,
        'action_var_schedule' : [1,0],
        'seed' : seed,
        'num_epochs' : 500
    }


    run_name = "env3_" + str(seed)
    p = Process(target=run_sg, args=(arg_dict, ppo, run_name, 'what is life?', "/data/cartpole5/"))
    p.start()
    proc_list.append(p)

    
for p in proc_list:
    print("joining")
    p.join()
