import torch.nn as nn
from seagul.rl.ppo.ppo2 import ppo
from seagul.nn import MLP
import torch
from seagul.rl.models import PPOModel
from multiprocessing import Process, Manager
import numpy as np
#torch.set_default_dtype(torch.double)


"""
Basic smoke test for PPO. This file contains an arg_dict that contains hyper parameters known to work with 
seagul's implementation of PPO. You can also run this script directly, which will check if the algorithm 
suceeds across 4 random seeds
Example:
from seagul.rl.regression_tests.pend_ppo2 import arg_dict
from seagul.rl.algos import ppo
t_model, rewards, var_dict = ppo(**arg_dict)  # Should get to -200 reward
"""


input_size = 3
output_size = 1
layer_size = 16
num_layers = 1
activation = nn.ReLU

policy = MLP(input_size, output_size, num_layers, layer_size, activation)
value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
model = PPOModel(policy, value_fn, action_var=0.1, discrete=False)

def run_and_test(arg_dict, retval):

    torch.set_num_threads(1)
    t_model, rewards, var_dict = ppo(**arg_dict)

    seed = arg_dict["seed"]
    if var_dict["early_stop"]:
        print("seed", seed, "achieved -200 reward in ", len(rewards), "steps")
        retval[seed] = True

    else:
        print("Error: seed:", seed, "failed")
        print("Rewards were", rewards[-1])
        retval[seed] = False

# Define our hyper parameters
arg_dict = {
    "env_name" : "Pendulum-v0",
    "total_steps" : 200*2048,
    "model" : model,
    "epoch_batch_size": 2048,  # how many steps we want to use before we update our gradients
    "reward_stop": -200,
    "pol_batch_size": 512,
    "val_batch_size": 512,
    "val_epochs": 10,
    "pol_epochs": 10,
    "pol_lr": 1e-2,
    "val_lr": 1e-3,
    "act_var_schedule": [0.707],
}

if __name__ == "__main__":

    proc_list = []
    manager = Manager()
    ret_dict = manager.dict()

    for seed in np.random.randint(0,2**32,8):
        arg_dict["seed"] = int(seed)
        p = Process(target=run_and_test, args=(arg_dict,ret_dict))
        p.start()
        proc_list.append(p)

    for p in proc_list:
        p.join()

print(ret_dict)
