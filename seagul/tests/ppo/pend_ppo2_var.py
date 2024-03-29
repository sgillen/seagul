import torch.nn as nn
from seagul.rl.ppo.ppo2 import PPOAgent
from seagul.nn import MLP
import torch
from seagul.rl.ppo.models import PPOModel
from multiprocessing import Process, Manager, Pool
import numpy as np
from seagul.plot import smooth_bounded_curve, chop_returns
import matplotlib.pyplot as plt


def run_and_test(seed, verbose=True):
    input_size = 3
    output_size = 1
    layer_size = 32
    num_layers = 2
    activation = nn.ReLU

    policy = MLP(input_size, output_size, num_layers, layer_size, activation)
    value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
    model = PPOModel(policy, value_fn, init_logstd=-.5, learn_std=True)

    # Define our hyper parameters
    agent = PPOAgent(env_name="Pendulum-v0", model=model, epoch_batch_size=2048, seed=int(seed), sgd_batch_size=64,
                     lr_schedule=(3e-4,), sgd_epochs=30, target_kl=float('inf'), clip_val=False, reward_stop=-200, env_no_term_steps=1000,
                     normalize_return=True, normalize_obs=True, normalize_adv=True)

    t_model, rewards, var_dict = agent.learn(total_steps = 2e6)

    if verbose:
        if var_dict["early_stop"]:
            print("seed", seed, "achieved target reward in ", len(rewards), "steps")
        else:
            print("Error: seed:", seed, "failed")

    return rewards, agent, var_dict


if __name__ == "__main__":
    seeds = np.random.randint(0,2**32,8)
    pool = Pool(processes=8)

    run_and_test(seeds[0])
    results = pool.map(run_and_test, seeds)

    rewards = []
    agents = []
    vars = []

    for result in results:
        rewards.append(result[0])
        agents.append(result[1])
        vars.append(result[2])

    # for reward in rewards:
    #     plt.plot(reward, alpha=.8)

    #rewards = np.array(rewards).transpose(1, 0)
    #smooth_bounded_curve(rewards, window=10)
    #print(finished)

    plt.show()
