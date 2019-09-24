import seagul.envs
import gym
gym.make('su_acro_drake-v0')

from seagul.rl.run_utils import run_and_save_bs
#from seagul.rl.algos import ppo, ppo_switch
#from seagul.rl.models import PpoModel, switchedPpoModel
#from seagul.nn import MLP, CategoricalMLP


from multiprocessing import Process
import seagul.envs

import time

import torch
import torch.nn as nn



## init policy, valuefn
input_size = 4
output_size = 1
layer_size = 24
num_layers=3
activation=nn.ReLU

torch.set_default_dtype(torch.double)
proc_list = []

for seed in range(4):

    #seed = int((time.time() % 1)*1e8)
    
    arg_dict = {
        'env': 'su_acro_drake-v0',
        'alg': 'ppo2',
        'network': 'mlp',
        'num_timesteps': '2e6',
        'num_env': '8',
        'num_layers': '2',
        'num_hidden': '24',
        'seed' :  str(seed)
    }


    run_name = "bs_ppo2" + str(seed)    
#    run_and_save_bs(arg_dict, run_name, 'baseline for ppo2', "/data/drake_base/")
    p = Process(target=run_and_save_bs, args=(arg_dict, run_name, 'baseline for ppo2', "/data/drake_base/"))
    p.start()
    p.join()
    
# for seed in range(6,10):

#     #seed = int((time.time() % 1)*1e8)
    
#     arg_dict = {
#         'env': 'su_acrobot-v0',
#         'alg': 'ppo2',
#         'network': 'mlp',
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
