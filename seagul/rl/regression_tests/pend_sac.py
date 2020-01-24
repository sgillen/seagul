import gym
import torch.nn as nn
from seagul.rl.algos.sac import sac
from seagul.nn import MLP, RBF
import torch
from seagul.rl.models import SACModel
from multiprocessing import Process
import time
start_time = time.time()

env_name = "Pendulum-v0"
input_size = 3
output_size = 1
layer_size = 64
num_layers = 2
activation = nn.ReLU

# policy = MLP(input_size, output_size * 2, num_layers, layer_size, activation)
# value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
# q1_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)
# q2_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)

policy = RBF(input_size, output_size * 2, layer_size)
value_fn = RBF(input_size, 1, layer_size)
q1_fn = RBF(input_size + output_size, 1, layer_size)
q2_fn = RBF(input_size + output_size, 1, layer_size)

model = SACModel(policy, value_fn, q1_fn, q2_fn, 1)

def run_and_test(arg_dict, seed):
    t_model, rewards, var_dict = sac("Pendulum-v0", 20000, model, seed=seed, **arg_dict)

    if var_dict["early_stop"]:
        print("seed", seed, "achieved 200 reward in ", len(rewards), "steps")
    else:
        print("Error: seed:", seed, "failed")
        print("Rewards were", rewards)
    return


# Define our hyper parameters
arg_dict = {"reward_stop": -200}

run_and_test(arg_dict, 0)
print("--- %s seconds ---" % (time.time() - start_time))

# proc_list = []
# for seed in [0, 1, 2, 3]:
#     p = Process(target=run_and_test, args=(arg_dict, seed))
#     p.start()
#     proc_list.append(p)

# for p in proc_list:
#     print("joining")
#     p.join()
