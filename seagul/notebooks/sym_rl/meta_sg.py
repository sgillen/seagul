from multiprocessing import Process
import seagul.envs

#import time
import gym

# Do this first because otherwise drake can break
env_name = 'Pendulum-v0'
env = gym.make(env_name)

import torch
import torch.nn as nn

from seagul.rl.run_utils import run_sg
from seagul.rl.algos import ppo
from seagul.rl.models import PpoModel, PpoModelActHold
from seagul.nn import MLP, CategoricalMLP



input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]
layer_size = 64
num_layers=2
activation=nn.ReLU

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
        value_fn=MLP(input_size, output_size, num_layers, layer_size, activation),
        discrete=False,
    )

    arg_dict = {
        'env_name' : env_name,
        'model' : model,
        'action_var_schedule' : [.5,.5],
        'seed' : seed, #int((time.time() % 1)*1e8),
        'num_epochs' : 300,
        'epoch_batch_size': 2048,
        'gamma' : .95,
        'p_epochs' : 32,
        'v_epochs' : 32,
        'policy_batch_size' : 64,
        'value_batch_size' : 64
    }
    
    run_name = "state_norm" + str(seed)

    
    run_sg(arg_dict, ppo, run_name, '', "/data/sg_pend2/")
    
    #p = Process(target=run_sg, args=(arg_dict, ppo, run_name, 'ppo for walker with nn policy, state norm turned on', "/data/linear_ppo/"))
    #p.start()
    #proc_list.append(p)

    
#for p in proc_list:
#    print("joining")
#    p.join()
