# Do this first because otherwise drake can break
import gym
import torch
import torch.nn as nn

from seagul.rl.run_utils import run_sg
from seagul.rl.algos import sac
from seagil.rl.algos.sac_sym import sac_sym
from seagul.nn import MLP
from seagul.rl.models import SACModel

env_name = "Pendulum-v0"
env = gym.make(env_name)

input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]
layer_size = 32
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
    'total_steps': 10000,
    'model' : model,
    'env_name': env_name,
    'reward_stop': -200,
    'seed': 2,
}

run_sg(arg_dict, sac, "sac_test", "testing sac saving", "/data/test/")




