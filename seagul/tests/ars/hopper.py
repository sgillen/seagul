import torch.nn as nn
import numpy as np
import gym
from seagul.rl.ars.ars_np_queue import ARSAgent
import matplotlib.pyplot as plt
import torch
import dill
import seagul.envs

#env_name = "hmap_hopper-v0"
env_name = "Hopper-v2"
env = gym.make(env_name)
max_reward = 3000


def run_and_test(seed, verbose=False):
    agent = ARSAgent(env_name=env_name, seed=int(seed), n_workers=8,
                     step_size=.05,
                     exp_noise=.08,
                     reward_stop=max_reward)
    model, rewards, var_dict,  = agent.learn(int(100))

    if verbose:
        if var_dict["early_stop"]:
            print("seed", seed, f"achieved {max_reward} reward in ", len(rewards), "steps")
        else:
            print("Error: seed:", seed, "failed")

    return rewards, agent, var_dict


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    seeds = np.random.randint(0, 2**32, 1)
    finished_list = []

    rewards, agent, local_dict = run_and_test(seeds[0])

