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
    input_size = 4
    output_size = 1
    layer_size = 16
    num_layers = 1
    activation = nn.ReLU

    policy = MLP(input_size, output_size, num_layers, layer_size, activation)
    value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
    model = PPOModel(policy, value_fn, init_logstd=-.5, learn_std=True)

    agent = PPOAgent(env_name="InvertedPendulum-v2",
                     model=model,
                     epoch_batch_size=2048,
                     reward_stop=1000,
                     sgd_batch_size=512,
                     sgd_epochs=50,
                     lr_schedule=(1e-3,),
                     target_kl=.05,
                     env_no_term_steps=1000,
                     normalize_adv=True,
                     normalize_return=False,
                     seed=int(seed))

    t_model, rewards, var_dict = agent.learn(total_steps = 2e6)

    if verbose:
        if var_dict["early_stop"]:
            print("seed", seed, "achieved 1000 reward in ", len(rewards), "steps")
        else:
            print("Error: seed:", seed, "failed")

    return rewards, var_dict["early_stop"]


if __name__ == "__main__":

    seeds = np.random.randint(0,2**32,8)

    #run_and_test(seeds[0])
    pool = Pool(processes=8)
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
