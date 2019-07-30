import torch.nn as nn
from seagul.rl.ppo2 import ppo
from seagul.rl.policies import Categorical_MLP, MLP
import torch

torch.set_default_dtype(torch.double)

policy = Categorical_MLP(input_size=4, output_size=1, layer_size=12, num_layers=2, activation=nn.ReLU)
value_fn = MLP(input_size=4, output_size=1, layer_size=12, num_layers=2, activation=nn.ReLU)

# Define our hyper parameters
num_epochs = 100
batch_size = 2048  # how many steps we want to use before we update our gradients
num_steps = 1000 # number of steps in an episode (unless we terminate early)
max_reward = num_steps
p_batch_size = 1024
v_epochs = 1
p_epochs = 10
p_lr = 1e-2
v_lr = 1e-2

gamma = .99
lam = .99
eps = .2

variance = 0.2 # feel like there should be a better way to do this...

env, model, value_fn = ppo('CartPole-v0', 500, policy, value_fn)