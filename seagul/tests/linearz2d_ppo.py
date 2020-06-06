import dill
import torch.nn as nn
from seagul.rl.ppo.ppo2 import ppo
from seagul.nn import MLP
import torch
from seagul.rl.ppo.models import PPOModel
from multiprocessing import Process, Manager, Pool
import numpy as np
from seagul.plot import smooth_bounded_curve, chop_returns
import matplotlib.pyplot as plt
from seagul.integration import euler
import seagul.envs


"""
Basic smoke test for PPO. This file contains an arg_dict that contains hyper parameters known to work with 
seagul's implementation of PPO. You can also run this script directly, which will check if the algorithm 
suceeds across 4 random seeds
Example:
from seagul.rl.tests.pend_ppo2 import arg_dict
from seagul.rl.algos import ppo
t_model, rewards, var_dict = ppo(**arg_dict)  # Should get to -200 reward
"""


def run_and_test(seed):
    alg_config["seed"] = int(seed)
    t_model, rewards, var_dict = ppo(**alg_config)

    seed = alg_config["seed"]
    if var_dict["early_stop"]:
        print("seed", seed, "achieved 1000 reward in ", len(rewards), "steps")
    else:
        print("Error: seed:", seed, "failed")

    with open("./tmp/workspace_" + str(seed), "wb") as outfile:
        torch.save(var_dict, outfile, pickle_module=dill)

    return rewards


def reward_fn(s):
    if s[2] > 0:
        if s[0] >= 0 and s[1] >= 0:
            reward = np.clip(np.sqrt(s[0]**2 + s[1]**2),0,10)
            #reward = 5 - np.clip(np.abs(np.sqrt(s[0]**2 + s[2]**2) - 5)**2,0,5)
            s[2] = -10
        else:
            reward = 0.0

    elif s[2] < 0:
        if s[0] <= 0 and s[1] <= 0:
            reward = np.clip(np.sqrt(s[0]**2 + s[1]**2),0,10)
            #reward = 5 - np.clip(np.abs(np.sqrt(s[0]**2 + s[2]**2)**2 - 5),0,5)
            s[2] = 10
        else:
            reward = 0.0

    return reward, s
#


if __name__ == "__main__":
    input_size = 3
    output_size = 1
    layer_size = 32
    num_layers = 2
    activation = nn.ReLU

    policy = MLP(input_size, output_size * 2, num_layers, layer_size, activation)
    value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
    model = PPOModel(policy, value_fn, action_std=0.1, fixed_std=False)
    env_name = "linear_z2d-v0"

    num_steps = 500
    env_config = {
        "reward_fn": reward_fn,
        "xz_max": float('inf'),
        "num_steps": num_steps,
        "act_hold": 10,
        "integrator": euler,
        "dt": .01,
        "init_noise_max": 10,
    }

    alg_config = {
        "env_name": env_name,
        "model": model,
        "total_steps": 2e6,
        "epoch_batch_size": 1024,
        "sgd_batch_size": 512,
        "lam": .2,
        "gamma": .95,
        "env_config": env_config,
        "sgd_epochs": 30,
        "reward_stop": 300
    }

    seeds = np.random.randint(0,2**32,8)
    pool = Pool(processes=8)
    results = pool.map(run_and_test, seeds)
    #results = run_and_test(seeds[0])
    results = chop_returns(results)
    results = np.array(results).transpose(1,0)

    smooth_bounded_curve(results)
    plt.show()

