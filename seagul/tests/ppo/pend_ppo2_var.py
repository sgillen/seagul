import torch.nn as nn
from seagul.rl.ppo.ppo2 import PPOAgent
from seagul.nn import MLP
import torch
from seagul.rl.ppo.models import PPOModel
from multiprocessing import Process, Manager, Pool
import numpy as np
from seagul.plot import smooth_bounded_curve, chop_returns
import matplotlib.pyplot as plt


"""
Basic smoke test for PPO. This file contains an arg_dict that contains hyper parameters known to work with 
seagul's implementation of PPO. You can also run this script directly, which will check if the algorithm 
suceeds across 4 random seeds
Example:
from seagul.rl.tests.pend_ppo2 import arg_dict
from seagul.rl.algos import ppo
t_model, rewards, var_dict = ppo(**arg_dict)  # Should get to -200 reward
"""


def run_and_test(seed, verbose=True):
    input_size = 3
    output_size = 1
    layer_size = 16
    num_layers = 1
    activation = nn.ReLU

    policy = MLP(input_size, output_size*2, num_layers, layer_size, activation)
    value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
    model = PPOModel(policy, value_fn, log_action_std=-.5, fixed_std=False)

    # Define our hyper parameters
    agent = PPOAgent(env_name="Pendulum-v0",
                     model=model,
                     epoch_batch_size=4096,
                     reward_stop=-200,
                     sgd_batch_size=64,
                     sgd_epochs=50,
                     target_kl=.05,
                     lr_schedule=(1e-2,),
                     normalize_return=False,
                     normalize_obs=True,
                     normalize_adv=True,
                     seed=int(seed))

    t_model, rewards, var_dict = agent.learn(total_steps = 2e5)

    if verbose:
        if var_dict["early_stop"]:
            print("seed", seed, "achieved 1000 reward in ", len(rewards), "steps")
        else:
            print("Error: seed:", seed, "failed")

    return rewards, var_dict["early_stop"]


if __name__ == "__main__":
    seeds = np.random.randint(0,2**32,8)
    pool = Pool(processes=8)
    results = pool.map(run_and_test, seeds)

    rewards = []
    finished = []
    for result in results:
        rewards.append(result[0])
        finished.append(result[1])

    for reward in rewards:
        plt.plot(reward, alpha=.8)

    #rewards = np.array(rewards).transpose(1, 0)
    #smooth_bounded_curve(rewards, window=10)
    print(finished)

    plt.show()
