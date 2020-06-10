import torch.nn as nn
from seagul.rl.sac.sac import sac
from seagul.nn import RBF, MLP
import torch
from seagul.rl.sac import SACModel
import time
from multiprocessing import Process, Manager
import numpy as np

"""
Basic smoke test for SAC. This file contains an arg_dict that contains hyper parameters known to work with 
seagul's implementation of SAC. You can also run this script directly, which will check if the algorithm 
suceeds across 4 random seeds

Example:

from seagul.rl.tests.pend_sac import arg_dict
from seagul.rl.algos import sac 
t_model, rewards, var_dict = sac(**arg_dict)  # Should get to -200 reward

"""


def run_and_test(arg_dict):
    torch.set_num_threads(1)

    t_model, rewards, var_dict = sac(**arg_dict)

    seed = arg_dict["seed"]
    if var_dict["early_stop"]:
        print("seed", seed, "achieved 200 reward in ", len(rewards), "steps")
    else:
        print("Error: seed:", seed, "failed")
        print("Rewards were", rewards[-1])


if __name__ == "__main__" :
    start_time = time.time()

    input_size = 3
    output_size = 1
    layer_size = 16
    num_layers = 2
    activation = nn.ReLU

    model = SACModel(
         policy = MLP(input_size, output_size * 2, num_layers, layer_size, activation),
         value_fn = MLP(input_size, 1, num_layers, layer_size, activation),
         q1_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation),
         q2_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation),
         act_limit=1
    )

    # Define our hyper parameters
    alg_config = {
        "env_name": "Pendulum-v0",
        "model": model,
        "exploration_steps" : 5000,
        "train_steps": 200000,
        "reward_stop": -200,
        "use_gpu": False,
    }

    # for debugging
    #alg_config["seed"] = 0
    #run_and_test(alg_config)

    proc_list = []

    for seed in np.random.randint(0,2**31,8):
        alg_config["seed"] = int(seed)
        p = Process(target=run_and_test, args=[alg_config])
        p.start()
        proc_list.append(p)

    for p in proc_list:
        p.join()

print(f"Total time: {(time.time() - start_time)}")
