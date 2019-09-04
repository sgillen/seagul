from seagul.rl.run_utils import run_sg
from seagul.rl.algos import ppo
from seagul.rl.models import ppoModel
from seagul.nn import MLP

import torch
import torch.nn as nn

## init policy, valuefn
input_size = 6
output_size = 1
layer_size = 64
num_layers=3
activation=nn.ReLU

torch.set_default_dtype(torch.double)

model = ppoModel(
    policy = MLP(input_size, output_size, num_layers, layer_size, activation),
    value_fn = MLP(input_size, 1, num_layers, layer_size, activation),
    action_var = 4
)

arg_dict = {
    'env_name' : 'su_acrobot-v0',
    'model' : model,
    'num_epochs' : 100,
    'action_var_schedule' : [10,0]
}

run_sg(arg_dict, ppo, run_name="debug")
