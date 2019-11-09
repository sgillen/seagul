from multiprocessing import Process
import seagul.envs

#import time

import gym


env_name = 'CartPole-v0'
env = gym.make(env_name)


import torch
import torch.nn as nn


#init policy, valuefn
input_size = env.observation_space.shape[0]
output_size = 1
layer_size = 12
num_layers= 2
activation=nn.ReLU

from seagul.rl.run_utils import run_sg, run_and_save_bs
from seagul.rl.algos import ppo, ppo_sym
from seagul.rl.models import PpoModel, switchedPpoModel, PpoModelActHold
from seagul.nn import MLP, CategoricalMLP

torch.set_default_dtype(torch.double)
proc_list = []


for seed in [0]:
    
    policy = MLP(input_size, output_size, num_layers, layer_size, activation)


    # model = PpoModelActHold(
    #     policy=policy,
    #     value_fn=MLP(input_size, 1, num_layers, layer_size, activation),
    #     discrete=False,
    #     hold_count = 200
    # )

    model = PpoModel(
        policy=policy,
        value_fn=MLP(input_size, 1, num_layers, layer_size, activation),
        discrete=True,
    )

    

    arg_dict = {
        'env_name' : env_name,
        'model' : model,
        'action_var_schedule' : [1,1],
        'seed' : seed, #int((time.time() % 1)*1e8),
        'num_epochs' : 1000,
        'epoch_batch_size': 2048,
        'gamma' : 1,
        'p_epochs' : 10,
        'v_epochs' : 10,
    }
    
    run_name = "ppo_sym" + str(seed)

    
#    run_sg(arg_dict, ppo, run_name, 'run with 100 epochs, torque limit', "/data/drake_acro_final/")
    
    p = Process(target=run_sg, args=(arg_dict, ppo, run_name, 'ppo for walker with nn policy, state norm turned on', "/data/cartpole/"))
    p.start()
    proc_list.append(p)

    
for p in proc_list:
    print("joining")
    p.join()
