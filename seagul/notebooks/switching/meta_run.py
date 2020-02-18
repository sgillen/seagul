
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

for max_torque in [1,5,25,100]:
    for seed in [0,1]:

        policy = MLP(input_size, output_size, num_layers, layer_size, activation)
        value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
        model = PPOModelActHold(
            policy=policy,
            value_fn = value_fn,
            discrete=False,
            hold_count = 0
        )

        def reward_fn(ns, act):
            reward = -(np.cos(ns[0]) + np.cos(ns[0] + ns[1]))
            reward -= (abs(ns[2]) > 5)
            reward -= (abs(ns[3]) > 10)
            return 1e-2*np.array(reward, np.float32)

        env_config = {
            "max_torque" : max_torque,
            "init_state" : [0.0, 0.0, 0.0, 0.0],
            "init_state_weights" : np.array([1, 1, 0, 0]),
            "dt" : .01,
            "max_t" : 2,
            "act_hold" : 5,
            "fixed_step" : True,
            "int_accuracy" : .01,
            "reward_fn" : reward_fn
        }

        alg_config = {
            "env_name": env_name,
            "model": model,
            "act_var_schedule": [1],
            "seed": seed,  # int((time.time() % 1)*1e8),
            "total_steps" : 500*2048,
            "epoch_batch_size": 2048,
            "reward_stop" : 900,
            "gamma": 1,
            "pol_epochs": 30,
            "val_epochs": 30,
            "env_config" : env_config
        }

        #    run_sg(alg_config, ppo, "debug_2pol" + str(seed), "debugging the large pol loss spike", "/data/debug")a
        now = datetime.datetime.now()
        date_str = str(now.day) + "-" + str(now.month) + "_" + str(now.hour) + "-" + str(now.minute)
        run_name = "small_rew" + str(seed) + "_" + str(max_torque) + "--" + date_str

        p = Process(target=run_sg, args=(alg_config, ppo, run_name , "debugging the large pol loss spike", "/data/small_rew_search3/"))
        p.start()
        proc_list.append(p)


for p in proc_list:
    p.join()

