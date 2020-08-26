from multiprocessing import Pool
import torch.nn as nn
import numpy as np
import gym
from seagul.rl.sac import SACAgent, SACModel
from seagul.nn import MLP
import matplotlib.pyplot as plt
import torch
import dill

env_name = "HalfCheetah-v2"

env = gym.make(env_name)


def run_and_test(seed, verbose=False):
    # init policy, value fn
    input_size = 17
    output_size = 6
    layer_size = 64
    num_layers = 2
    activation = nn.ReLU

    model = SACModel(
         policy = MLP(input_size, output_size*2, num_layers, layer_size, activation),
         value_fn= MLP(input_size, 1, num_layers, layer_size, activation),
         q1_fn = MLP(input_size+output_size, 1, num_layers, layer_size, activation),
         q2_fn = MLP(input_size+output_size, 1, num_layers, layer_size, activation),
         act_limit=3
    )

    agent = SACAgent(env_name=env_name, model=model, seed=int(seed), exploration_steps=5000,
                     min_steps_per_update=500, reward_stop=3000, gamma=1, sgd_batch_size=64,
                     replay_batch_size =2048, iters_per_update=1000, env_max_steps=1000,
                     polyak=.995)

    model, rewards, var_dict,  = agent.learn(train_steps=1e6)
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
