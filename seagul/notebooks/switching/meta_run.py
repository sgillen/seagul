import seagul.envs
import gym

env_name = "su_acrobot-v0"
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
layer_size = 16
num_layers = 1
activation = nn.ReLU


proc_list = []
trial_num = input("What trial is this?\n")

for seed in np.random.randint(0, 2**32, 4):
    for max_t in [5,10,20]:

        max_torque = 5
        max_t = max_t

        policy = MLP(input_size, output_size, num_layers, layer_size, activation)
        value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
        model = PPOModelActHold(
            policy=policy,
            value_fn = value_fn,
            discrete=False,
            hold_count = 20
        )

        def reward_fn(ns, act):
            #return -1e-4*(5*(ns[0] - np.pi)**2 + ns[1]**2 + .5*ns[2]**2 + .5*ns[3]**2)
            return -1e-2*(np.cos(ns[0]) + np.cos(ns[0] + ns[1]))
            #return -.1 * np.exp(np.sqrt(.1 * (ns[0] - np.pi) ** 2 + .1 * ns[1] ** 2 + .01 * ns[2] ** 2 + .01 * ns[3] ** 2))


        env_config = {
            "init_state": [0, 0, 0, 0],
            "max_torque": max_torque,
            "init_state_weights": [0, 0, 0, 0],
            "dt": .01,
            "reward_fn" : reward_fn,
            "max_t" : max_t
            }

        alg_config = {
            "env_name": env_name,
            "model": model,
            "act_var_schedule": [3],
            "seed": 0,  # int((time.time() % 1)*1e8),
            "total_steps" : 5e5,
            "epoch_batch_size": 2048,
            "reward_stop" : None,
            "gamma": 1,
            "pol_epochs": 30,
            "val_epochs": 10,
            "env_config" : env_config
        }

        run_name = "swingup" + str(seed)

        p = Process(target=run_sg, args=(alg_config, ppo, run_name , "normal reward", "/data/sg_acro2/trial" + str(trial_num) + "_maxt" + str(max_t) + "/"))
        p.start()
        proc_list.append(p)


for p in proc_list:
    p.join()

