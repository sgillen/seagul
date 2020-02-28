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
    for max_torque in [5.0, 10.0]:
        max_torque = max_torque
        act_var = 3.0

        
        max_t = 10.0 

        policy = MLP(input_size, output_size, num_layers, layer_size, activation)
        value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
        model = PPOModel(
            policy=policy,
            value_fn=value_fn,
            discrete=False,
            )

        def reward_fn(ns, act):
            reward = 1*(np.sin(ns[0]) + np.sin(ns[0] + ns[1]))

            done = False
            if abs(ns[0] - np.pi/2) < 1 and abs(ns[1]) < .2 and abs(ns[2]) < 3 and abs(ns[3]) < 3:
                #reward += 20
                print("go zone")
                #done = True

            return reward, done

        m1 = 1;
        m2 = 1;
        l1 = 1;
        l2 = 2;
        lc1 = .5;
        lc2 = 1;
        I1 = .083;
        I2 = .33;
        g = 9.8;

        env_config = {
            "init_state": [0, 0, 0, 0],
            "max_torque": max_torque,
            "init_state_weights": [0, 0, 0, 0],
            "dt": .01,
            "reward_fn" : reward_fn,
            "max_t" : max_t,
            "m2": m2,
            "m1": m1,
            "l1": l1,
            "lc1": lc1,
            "lc2": lc2,
            "i1": I1,
            "i2": I2,
            "act_hold" : 20
        }

        alg_config = {
            "env_name": env_name,
            "model": model,
            "act_var_schedule": [act_var],
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

        p = Process(target=run_sg, args=(alg_config, ppo, run_name , "normal reward", "/data3/sg_acro_kb/trial" + str(trial_num) + "_t" + str(max_torque) + "/"))
        p.start()
        proc_list.append(p)


for p in proc_list:
    p.join()

