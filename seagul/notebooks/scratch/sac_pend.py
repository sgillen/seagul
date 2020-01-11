from seagul.rl.algos import ppo
from seagul.nn import MLP
from seagul.rl.models import PPOModel
import torch

torch.set_default_dtype(torch.double)# TODO need to update everything to support arbitrary dtypes

input_size = 3
output_size = 1
layer_size = 64
num_layers = 2

policy = MLP(input_size, output_size, num_layers, layer_size)
value_fn = MLP(input_size, 1, num_layers, layer_size)
model = PPOModel(policy, value_fn)

model, rews, var_dict = ppo("Pendulum-v0", 10000, model)
