
import seagul.envs
import gym

env_name = "su_acro_drake-v0"
env = gym.make(env_name)

import datetime

import torch
import torch.nn as nn
import numpy as np
from multiprocessing import Process
from seagul.rl.run_utils import run_sg, run_and_save_bs
from seagul.rl.algos import ppo, ppo_switch
from seagul.rl.models import PPOModel, SwitchedPPOModel, PPOModelActHold
from seagul.nn import MLP, CategoricalMLP
import time

# init policy, valuefn
input_size = 4
output_size = 1
layer_size = 24
num_layers = 2
activation = nn.ReLU


proc_list = []

for seed in [0,1,2,3]:
        policy = MLP(input_size, output_size, num_layers, layer_size, activation)
        value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
        model = PPOModelActHold(
            policy=policy,
            value_fn = value_fn,
            discrete=False,
            hold_count = 0
        )

        def reward_fn(ns, act):
            return -1e-4*(ns[0]**2 + ns[1]**2 + .1*ns[2]**2 + .2*ns[3]**2)
            #return 1e-2*(np.cos(ns[0]) + np.cos(ns[0] + ns[1]))

        env_config = {
            "max_torque" : 25,
            "init_state" : [0.0, 0.0, 0.0, 0.0],
            "init_state_weights" : np.array([1, 1, 0, 0]),
            "dt" : .01,
            "max_t" : 10,
            "act_hold" : 1,
            "fixed_step" : True,
            "reward_fn" : reward_fn,
            # "max_th1dot" : 20,
            # "max_th2dot" : 40
        }

        alg_config = {
            "env_name": env_name,
            "model": model,
            "act_var_schedule": [1],
            "seed": seed,  # int((time.time() % 1)*1e8),
            "total_steps" : 5e5,
            "epoch_batch_size": 2048,
            "reward_stop" : None,
            "gamma": 1,
            "pol_epochs": 10,
            "val_epochs": 10,
            "env_config" : env_config
        }

        run_name = "swingdown" + str(seed)

        p = Process(target=run_sg, args=(alg_config, ppo, run_name , "new reward, just the states now with angle wrap", "/data_sd/trial7/"))
        p.start()
        proc_list.append(p)


for p in proc_list:
    p.join()

