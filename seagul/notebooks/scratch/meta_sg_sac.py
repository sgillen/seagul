import gym
import torch
import torch.nn as nn

from seagul.rl.run_utils import run_sg
from seagul.rl.algos import sac
from seagul.nn import MLP
from seagul.rl.models import SACModel

env_name = "Walker2d-v2"
env = gym.make(env_name)

input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]
layer_size = 128
num_layers = 2
activation = nn.ReLU

proc_list = []


policy = MLP(input_size, output_size * 2, num_layers, layer_size, activation)

# Do I need to do weight sharing here?
value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
q1_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)
q2_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)
model = SACModel(policy, value_fn, q1_fn, q2_fn, 1)


arg_dict = {
    'total_steps': 1.2e4,
    'model' : model,
    'env_name': env_name,
    'seed': 2,
    'env_steps' : 1000,
    'iters_per_update' : 3000,
    'min_steps_per_update': 1000,
    'reward_stop' : 3000, 
    'exploration_steps' : 10000,
    'replay_batch_size' : 100, 
    'use_gpu':False,
}

run_sg(arg_dict, sac, "/sac_walker0", "trying to get walker to work at all", "/sac_walker")
