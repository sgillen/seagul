from multiprocessing import Process
import torch.nn as nn
import numpy as np
import gym
from seagul.rl.sac import sac, SACModel
from seagul.nn import MLP
from seagul.rl.run_utils import run_sg
import time
import torch

proc_list = []
env_name = "InvertedPendulum-v2"

env = gym.make(env_name)


def run_and_test(arg_dict):
    torch.set_num_threads(1)

    t_model, rewards, var_dict = sac(**arg_dict)

    seed = arg_dict["seed"]
    if var_dict["early_stop"]:
        print("seed", seed, "achieved max reward in ", len(rewards), "steps")
    else:
        print("Error: seed:", seed, "failed")
        print("Rewards were", rewards[-1])


start = time.time()
for seed in np.random.randint(0, 2 ** 32, 8):
    # init policy, value fn
    input_size = 4
    output_size = 1
    layer_size = 16
    num_layers = 1
    activation = nn.ReLU

    model = SACModel(
         policy = MLP(input_size, output_size*2, num_layers, layer_size, activation),
         value_fn= MLP(input_size, input_size, num_layers, layer_size, activation),
         q1_fn = MLP(input_size+output_size, 1, num_layers, layer_size, activation),
         q2_fn = MLP(input_size+output_size, 1, num_layers, layer_size, activation),
         act_limit=3
    )

    alg_config = {
        "env_name": env_name,
        "model": model,
        "seed": int(seed),  # int((time.time() % 1)*1e8),
        "train_steps" : 1e6,
        "exploration_steps" : 50000,
        "min_steps_per_update" : 500,
        "reward_stop" : 1000,
        "gamma": 1,
        "sgd_batch_size": 64,
        "replay_batch_size" : 2048,
        "iters_per_update": 1000,
        "env_max_steps":1000,
        "polyak":.995
        #"iters_per_update": float('inf'),
    }


    run_sg(alg_config, sac, "sac bullet defaults", "debug", "/data/" + "/" + "seed" + str(seed))

    p = Process(target=run_and_test, args=[alg_config])
    p.start()
    proc_list.append(p)

for p in proc_list:
    p.join()

print(f"Total time: {(time.time() - start)}")

