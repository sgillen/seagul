import gym
import seagul.envs

import seagul.envs
import torch.nn as nn
import numpy as np
from numpy import pi
from multiprocessing import Process
from seagul.rl.run_utils import run_sg
from seagul.rl.ppo import ppo, PPOModel
from seagul.nn import MLP
import torch

# init policy, valuefn
input_size = 8
output_size = 1
layer_size = 32
num_layers = 1
activation = nn.ReLU

proc_list = []
trial_num = input("What trial is this??\n")

env_name = "bball-v0"

for seed in np.random.randint(0, 2**32, 4):

            model = PPOModel(
                policy=MLP(input_size, output_size*2, layer_size, num_layers),
                value_fn=MLP(input_size, output_size, layer_size, num_layers),
                fixed_std=False
            )

            alg_config = {
                "env_name": env_name,
                "model": model,
                "seed": int(seed),  # int((time.time() % 1)*1e8),
                "total_steps" : 5e5,
                "reward_stop" : None,
                "gamma": 1,
                "pol_epochs": 30,
                "val_epochs": 30,
                "env_config" : {},
                }

            run_name = "swingup" + str(seed)

            run_sg(alg_config, ppo, run_name , "normal reward", "/data4/switching_sortof/trial" + str(trial_num)  + "/")
            #p = Process(target=run_sg, args=(alg_config, ppo, run_name , "warm_start", "/data5/acro2/trial" + str(trial_num)  + "/"))
            #p.start()
            #proc_list.append(p)


for p in proc_list:
    p.join()