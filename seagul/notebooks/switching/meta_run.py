from seagul.rl.run_utils import run_sg, run_and_save_bs
from seagul.rl.algos import ppo, ppo_switch
from seagul.rl.models import ppoModel, switchedPpoModel
from seagul.nn import MLP
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

for seed in range(8):

    env_name = 'su_cartpole-v0'
    env = gym.make(env_name)
    
    
    model = switchedPpoModel(
        policy = MLP(input_size, output_size, num_layers, layer_size, activation),
        value_fn = MLP(input_size, 1, num_layers, layer_size, activation),
        gate_fn  = torch.load("gate_fn"),
        nominal_policy=LQRControl,
        env=env
    )
    
    
    arg_dict = {
        'env_name' : env_name,
        'model' : model,
        'num_epochs' : 500,
        'action_var_schedule' : [10,0],
        'gate_var_schedule'   : [1,0],
    }


    run_name = "sw_ppo_warm" + str(seed)
    p = Process(target=run_sg, args=(arg_dict, ppo_switch, run_name, '', "/data/mp_test/"))
    p.start()
    proc_list.append(p)

for p in proc_list:
    print("joining")
    p.join()
#run_sg(arg_dict, ppo, run_name = run_name, run_desc = '', base_path="/data/cartpole/")


#    run_sg(arg_dict, ppo_switch, base_path="/data/acrobot/")
    
    
    


    # model = ppoModel(
    #     policy=MLP(input_size, output_size, num_layers, layer_size, activation),
    #     value_fn=MLP(input_size, 1, num_layers, layer_size, activation),
    # )

    # arg_dict = {
    #     'env_name' : 'su_cartpole-v0',
    #     'model' : model,
    #     'action_var_schedule' : [10,0],
    #     'seed' : seed,
    #     'num_epochs' : 50
    # }

    



# proc_list = []


#     arg_dict = {
#         'env': 'su_acrobot-v0',
#         'alg': 'ppo2',
#         'network': 'mlp',
#         'num_timesteps': '5e6',
#         'num_env': '1',
#         'num_layers': '3',
#         'num_hidden': '64',
#         'seed' : str(seed)
#     }
#     run_name = "ppo_acrobot_baseline" + str(seed)
#     p = Process(target=run_and_save_bs, args=  (arg_dict, run_name, '', '/data/acrobot/'))
#     p.start()
#     proc_list.append(p)

# for p in proc_list:
#     print("joining")
#     p.join()
#     # run_sg(arg_dict, ppo, run_name = run_name, run_desc = '', base_path="/data/cartpole/")


#run_baselines(arg_dict, run_name='discrete1', description='first run with discrete environment, appears to actually get positive rewards!')
