import gym
import seagul.envs

env_name = 'su_acro_drake-v0'
env = gym.make(env_name)


from seagul.rl.run_utils import run_sg, run_and_save_bs
from seagul.rl.algos import ppo, ppo_switch
from seagul.rl.models import PpoModel, switchedPpoModel, SwitchedPpoModelActHold
from seagul.nn import MLP, CategoricalMLP, DummyNet
from seagul.sims.cartpole import LQRControl


import torch
import torch.nn as nn


import numpy as np
from numpy import pi

from multiprocessing import Process

## init policy, valuefn
input_size = 4
output_size = 1
layer_size = 24
num_layers=3
activation=nn.ReLU

torch.set_default_dtype(torch.double)
proc_list = []

for seed in range(6,10):

    env_name = 'su_acro_drake-v0'
    env = gym.make(env_name)
    

    # # hard coded gate for debugging
    # def gate(state):
    #     if len(state.shape) == 1:
    #         return (((140 * pi / 180 < state[0] < pi) and state[1] <= 0) or (
    #                 (pi < state[0] < 220 * pi / 180) and state[1] >= 0))
    #     else:
    #         ret  = ((((140 * pi / 180 < state[:,0]) & (state[:,0] < pi)) & (state[:,1] <= 0))
    #                | ((pi < state[:,0]) & (state[:,0] < 220 * pi / 180) & (state[:,1] >= 0)))
    #         return torch.as_tensor(ret,dtype=torch.double).reshape(-1,1)


    
    # hard coded gate for debugging
    # def gate(state):
    #     if len(state.shape) == 1:
    #         return ((140 * pi / 180 < state[0] < pi) or (pi < state[0] < 220 * pi / 180))
    #     else:
    #         ret  = ( ((140 * pi / 180 < state[:,0]) & (state[:,0] < pi)) | ((pi < state[:,0]) & (state[:,0] < 220 * pi / 180)))
                      
    #         return torch.as_tensor(ret,dtype=torch.double).reshape(-1,1)


    
    #    gate_fn.net_fn = gate
    
    # def control(env,q):
    #     k = np.array([-1000, 1000, -10, -10])
    #     goal = np.copy(env.state)
    #     goal[0] -= pi
    #     return -k.dot(goal)

    def control(q):
        k = np.array([[278.44223126, 112.29125985, 119.72457377,  56.82824017]])
        gs = np.array([pi,0,0,0])
        #return 0
        return (-k.dot(gs - np.asarray(q))).squeeze()


    
    # model = SwitchedPpoModelActHold(
    #     #policy = MLP(input_size, output_size, num_layers, layer_size, activation),
    #     policy = torch.load("warm_policy_dr"),
    #     value_fn = torch.load("warm_value_dr"),
    #     #MLP(input_size, 1, num_layers, layer_size, activation),
    #     gate_fn  = torch.load("gate_fn_dr"),
    #     nominal_policy=control,
    #     hold_count = 200,
    # )

    
    model = switchedPpoModel(
        policy = torch.load("warm_policy_dr"),
        value_fn = torch.load("warm_value_dr"),
        gate_fn  = torch.load("gate_fn_dr"),
        # policy = MLP(input_size, output_size, num_layers, layer_size, activation),
        # value_fn = MLP(input_size, 1, num_layers, layer_size, activation),
        # gate_fn = CategoricalMLP(input_size, 1, num_layers, layer_size, activation), 
        nominal_policy=control,
        env=None
    )
        
    arg_dict = {
        'env_name' : env_name,
        'model' : model,
        'num_epochs' : 500,
        'epoch_batch_size': 2048,
        'action_var_schedule' : [1,1],
        'gate_var_schedule'   : [.3,.3],
        'gamma' : 1,
        'seed': seed
    }


    run_name = "500_30_" + str(seed)

    p = Process(target=run_sg,    args = (arg_dict, ppo_switch, run_name, 'saturate at 30N*M', "/data/drake_acro_switch2/"))
    p.start()
    proc_list.append(p)

for p in proc_list:
    print("joining")
    p.join()
