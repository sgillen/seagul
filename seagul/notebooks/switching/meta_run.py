from seagul.rl.run_utils import run_sg, run_and_save_bs
from seagul.rl.algos import ppo, ppo_switch
from seagul.rl.models import PpoModel, switchedPpoModel, PpoModelActHold
from seagul.nn import MLP, CategoricalMLP
from seagul.sims.cartpole import LQRControl
from multiprocessing import Process
import seagul.envs

import time

import torch
import torch.nn as nn

import gym

## init policy, valuefn
input_size = 6
output_size = 1
layer_size = 12
num_layers=2
activation=nn.Tanh

torch.set_default_dtype(torch.double)
proc_list = []

for seed in range(4):

    policy = MLP(input_size, output_size, num_layers, layer_size, activation)

    # model = PpoModelActHold(
    #     policy=policy,
    #     value_fn=MLP(input_size, 1, num_layers, layer_size, activation),
    #     discrete=False,
    #     hold_count = 200
    # )

    model = PpoModelActHold(
        policy=policy,
        value_fn=MLP(input_size, 1, num_layers, layer_size, activation),
        discrete=False,
        hold_count = 200
    )

    arg_dict = {
        'env_name' : 'su_acrobot-v0',
        'model' : model,
        'action_var_schedule' : [1,.001],
        'seed' : seed, #int((time.time() % 1)*1e8),
        'num_epochs' : 200,
        'gamma' : .99,
        'p_epochs' : 10,
        'v_epochs' : 10
        
    }
    run_name = "acrobot_116_" + str(seed)
    #run_sg(arg_dict, ppo, run_name, 'aaghhh', "/data/acrobot8/")

    p = Process(target=run_sg, args=(arg_dict, ppo, run_name, 'acrobot with gamma 1, more metrics. lower value loss', "/data/inv_pend/"))
    p.start()
    proc_list.append(p)

    
for p in proc_list:
    print("joining")
    p.join()
