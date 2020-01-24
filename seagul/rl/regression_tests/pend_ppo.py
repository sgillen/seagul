import torch.nn as nn
from seagul.rl.algos.ppo import ppo
from seagul.nn import MLP, CategoricalMLP, RBF
import torch
import matplotlib.pyplot as plt
from seagul.rl.models import PPOModel
from multiprocessing import Process

torch.set_default_dtype(torch.double)


input_size = 3
output_size = 1
layer_size = 64
num_layers = 2
activation = nn.ReLU

# policy = MLP(input_size, output_size, num_layers, layer_size, activation)
# value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
policy = RBF(input_size, output_size, layer_size)
value_fn = RBF(input_size, 1, layer_size)
model = PPOModel(policy, value_fn, action_var=0.1, discrete=False)


def run_and_test(arg_dict, seed):
    t_model, rewards, var_dict = ppo("Pendulum-v0", 200, model, seed=seed, **arg_dict)

    if len(rewards) >= 200:
        print("Error: seed:", seed, "failed")
    else:
        print("seed", seed, "achieved 200 reward in ", len(rewards), "steps")

    return


# Define our hyper parameters
arg_dict = {
    "epoch_batch_size": 2048,  # how many steps we want to use before we update our gradients
    "env_timesteps": 199,  # number of steps in an episode (unless we terminate early)
    "reward_stop": -200,
    "policy_batch_size": 512,
    "value_batch_size": 512,
    "v_epochs": 10,
    "p_epochs": 10,
    "policy_lr": 1e-2,
    "value_lr": 1e-2,
    "action_var_schedule": [0.707],
}


# proc_list = []
# for seed in [0, 1, 2, 3]:
#     p = Process(target=run_and_test, args=(arg_dict, seed))
#     p.start()
#     proc_list.append(p)


# for p in proc_list:
#     print("joining")
#     p.join()


t_model, rewards, var_dict = ppo("Pendulum-v0", 200, model, **arg_dict)
