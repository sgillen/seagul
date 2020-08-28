from multiprocessing import Pool
import torch.nn as nn
import numpy as np
import gym
from seagul.rl.ars.ars_pipe2 import ARSAgent
from seagul.nn import MLP
import matplotlib.pyplot as plt
import torch
import dill

env_name = "HalfCheetah-v2"
env = gym.make(env_name)


def run_and_test(seed, verbose=False):
    input_size = 17
    output_size = 6

    model = MLP(input_size, output_size, 0, 0)

    agent = ARSAgent(env_name=env_name, policy=model, seed=int(seed), n_workers=8, reward_stop=3000)
    model, rewards, var_dict,  = agent.learn(int(1e6))

    if verbose:
        if var_dict["early_stop"]:
            print("seed", seed, "achieved 1000 reward in ", len(rewards), "steps")
        else:
            print("Error: seed:", seed, "failed")

    return rewards, var_dict["early_stop"]


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    seeds = np.random.randint(0, 2**32, 8)
    finished_list = []

    for seed in seeds:
        rewards, finished = run_and_test(seed)
        finished_list.append(finished)

    print(finished_list)