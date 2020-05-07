import torch.nn as nn
from seagul.rl.ppo.ppo2 import ppo
from seagul.nn import MLP
import torch
from seagul.rl.ppo.models import PPOModel
from multiprocessing import Process, Manager
import numpy as np


"""
Basic smoke test for PPO. This file contains an arg_dict that contains hyper parameters known to work with 
seagul's implementation of PPO. You can also run this script directly, which will check if the algorithm 
suceeds across 4 random seeds
Example:
from seagul.rl.tests.pend_ppo2 import arg_dict
from seagul.rl.algos import ppo
t_model, rewards, var_dict = ppo(**arg_dict)  # Should get to -200 reward
"""

def run_and_test(arg_dict):

    t_model, rewards, var_dict = ppo(**arg_dict)

    seed = arg_dict["seed"]
    if var_dict["early_stop"]:
        print("seed", seed, "achieved 1000 reward in ", len(rewards), "steps")
    else:
        print("Error: seed:", seed, "failed")
        print("Rewards were", rewards[-1])

    return

if __name__ == "__main__":
    input_size = 4
    output_size = 1
    layer_size = 16
    num_layers = 1
    activation = nn.ReLU

    policy = MLP(input_size, output_size*2, num_layers, layer_size, activation)
    value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
    model = PPOModel(policy, value_fn, action_std=0.1, fixed_std=False)

    # Define our hyper parameters
    arg_dict = {
        "env_name" : "InvertedPendulum-v2",
        "total_steps" : 200*2048,
        "model" : model,
        "epoch_batch_size": 2048,  # how many steps we want to use before we update our gradients
        "reward_stop": 1000,
        "sgd_batch_size": 512,
        "val_epochs": 30,
        "pol_epochs": 30,
        "pol_lr": 1e-2,
        "val_lr": 1e-3,
        "env_no_term_steps": 1000
        }

    proc_list = []

    #run_and_test(arg_dict)

    for seed in np.random.randint(0, 2**32, 8):
        arg_dict["seed"] = int(seed)
        p = Process(target=run_and_test, args=[arg_dict])
        p.start()
        proc_list.append(p)

    for p in proc_list:
        p.join()

