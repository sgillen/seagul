
import seagul.envs
import gym

env_name = "su_acro_drake-v0"
env = gym.make(env_name)

import torch
import torch.nn as nn
import numpy as np
from multiprocessing import Process
from seagul.rl.run_utils import run_sg, run_and_save_bs
from seagul.rl.algos import ppo, ppo_switch
from seagul.rl.models import PPOModel, SwitchedPPOModel, PPOModelActHold
from seagul.nn import MLP, CategoricalMLP


# init policy, valuefn
input_size = 4
output_size = 1
layer_size = 24
num_layers = 2
activation = nn.ReLU


proc_list = []

for seed in [0,1,2,3]:
    torch.set_num_threads(1)
    proc_list = []
    
    policy = MLP(input_size, output_size, num_layers, layer_size, activation)
    value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
    model = PPOModelActHold(
        policy=policy,
        value_fn = value_fn,
        discrete=False,
        hold_count = 20
    )


    def reward_fn(ns):
        reward = -(np.cos(ns[0]) + np.cos(ns[0] + ns[1]))
        reward -= (abs(ns[2]) > 10)
        reward -= (abs(ns[3]) > 30) 
        return reward

    env_config = {
        "max_torque" : 10,
        "init_state" : [0.0, 0.0, 0.0, 0.0],
        "init_state_weights" : np.array([0, 0, 0, 0]),
        "dt" : .01,
        "max_t" : 5,
        "act_hold" : 1,
        "fixed_step" : True,
        "int_accuracy" : .01,
        "reward_fn" : reward_fn 
    }

    alg_config = {
        "env_name": env_name,
        "model": model,
        "act_var_schedule": [2],
        "seed": seed,  # int((time.time() % 1)*1e8),
        "total_steps" : 500*2048,
        "epoch_batch_size": 2048,
        "reward_stop" : 900,
        "gamma": 1,
        "pol_epochs": 10,
        "val_epochs": 10,
        "env_config" : env_config
    }

    #    run_sg(alg_config, ppo, "debug_2pol" + str(seed), "debugging the large pol loss spike", "/data/debug")a
    p  = Process(target=run_sg, args=(alg_config, ppo, "low_t" + str(seed), "debugging the large pol loss spike", "/data/low_t"))
    p.start()
    proc_list.append(p)


for p in proc_list:
    p.join()

