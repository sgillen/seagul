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
import torch
import dill

"""
Basic smoke test for PPO. This file contains an arg_dict that contains hyper parameters known to work with 
seagul's implementation of PPO. You can also run this script directly, which will check if the algorithm 
suceeds across 4 random seeds
Example:
from seagul.rl.tests.pend_ppo2 import arg_dict
from seagul.rl.algos import ppo
t_model, rewards, var_dict = ppo(**arg_dict)  # Should get to -200 reward
"""


def run_and_test(seed, verbose=False):
    input_size = 11
    output_size = 2
    layer_size = 32
    num_layers = 2
    activation = nn.ReLU

    policy = MLP(input_size, output_size * 2, num_layers, layer_size, activation)
    value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
    model = PPOModel(policy, value_fn, action_std=0.1, fixed_std=False)

    t_model, rewards, var_dict = ppo(env_name="Reacher-v2",
                                     total_steps=5e5,
                                     model=model,
                                     epoch_batch_size=2048,
                                     reward_stop=3000,
                                     sgd_batch_size=64,
                                     sgd_epochs=10,
                                     lr_schedule=[3e-4],
                                     target_kl=float('inf'),
                                     env_no_term_steps=50,
                                     entropy_coef=0.0,
                                     normalize_return=False,
                                     normalize_obs=True,
                                     normalize_adv=True,
                                     clip_val=False,
                                     seed=int(seed))


    torch.save(var_dict, open("./tmp/" + str(seed), 'wb'), pickle_module=dill)

    if verbose:
        if var_dict["early_stop"]:
            print("seed", seed, "achieved 1000 reward in ", len(rewards), "steps")
        else:
            print("Error: seed:", seed, "failed")

    return rewards, var_dict["early_stop"]


if __name__ == "__main__":
    seeds = np.random.randint(0, 2**32, 8)
    pool = Pool(processes=8)

    #results = run_and_test(run_and_test(seeds[0]))
    results = pool.map(run_and_test, seeds)

    rewards = []
    finished = []
    for result in results:
        rewards.append(result[0])
        finished.append(result[1])

    for reward in rewards:
        plt.plot(reward, alpha=.8)

    print(finished)

    plt.show()

    ws = torch.load(open(f'/home/sgillen/work/seagul/seagul/tests/ppo/todo/tmp/{seeds[0]}', 'rb'))
