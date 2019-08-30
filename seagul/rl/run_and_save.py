# ============================================================================================

import torch.nn as nn
from seagul.rl.ppo import ppo
from seagul.nn import Categorical_MLP, MLP, DummyNet
import torch
import time
import datetime
import json
import os
import errno

import matplotlib.pyplot as plt

torch.set_default_dtype(torch.double)

policy = MLP(input_size=4, output_size=1, layer_size=12, num_layers=2, activation=nn.ReLU)
value_fn = MLP(input_size=4, output_size=1, layer_size=12, num_layers=2, activation=nn.ReLU)
gate_fn = Categorical_MLP(input_size=4, output_size=1, layer_size=12, num_layers=2, activation=nn.ReLU)

# Define our hyper parameters
num_epochs = 100
batch_size = 2048  # how many steps we want to use before we update our gradients
num_steps = 1000  # number of steps in an episode (unless we terminate early)
max_reward = num_steps
p_batch_size = 2048
v_epochs = 1
p_epochs = 10
p_lr = 1e-2
v_lr = 1e-2

gamma = 0.99
lam = 0.99
eps = 0.2

variance = 0.2  # feel like there should be a better way to do this...

# env2, t_policy, t_val, rewards = ppo('InvertedPendulum-v2', 100, policy, value_fn)
start = time.time()
env2, t_policy, t_val, rewards, arg_dict = ppo("su_cartpole-v0", 1, policy, value_fn, epoch_batch_size=0)
end = time.time()
runtime = end - start

save_dir = "./data/test/"

os.makedirs(os.path.dirname(save_dir), exist_ok=True)

datetime_str = str(datetime.datetime.today())
datetime_str = datetime_str.replace(" ", "_")
runtime_str = str(datetime.timedelta(seconds=runtime))

 # This is the wrong way right??
torch.save(t_policy, open(save_dir + "policy", "wb"))
torch.save(t_val, open(save_dir + "value_fn", "wb"))

print(arg_dict)
description = "this is just a test2"
with open(save_dir + "info.json", "w") as outfile:
    json.dump(
        {
            "args": arg_dict,
            "metadata": {"date_time": datetime_str, "total runtime": runtime_str, "description": description},
        },
        outfile,
        indent=4,
    )
