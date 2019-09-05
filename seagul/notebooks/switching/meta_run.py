from seagul.rl.run_utils import run_sg, run_and_save_bs
from seagul.rl.algos import ppo, ppo_switch
from seagul.rl.models import ppoModel, switchedPpoModel
from seagul.nn import MLP
from seagul.sims.cartpole import LQRControl

import torch
import torch.nn as nn

import gym

## init policy, valuefn
input_size = 6
output_size = 1
layer_size = 64
num_layers=3
activation=nn.ReLU

torch.set_default_dtype(torch.double)

model = switchedPpoModel(
    policy = MLP(input_size, output_size, num_layers, layer_size, activation),
    value_fn = MLP(input_size, 1, num_layers, layer_size, activation),
    gate_fn =  MLP(input_size, 1, num_layers, layer_size, activation),
    nominal_policy=LQRControl
)

env_name = 'su_cartpole-v0',
env = gym.make(env_name)

arg_dict = {
    'env_name' : env_name,
    'env':env,
    'model' : model,
    'num_epochs' : 500,
    'action_var_schedule' : [10,0],
    'gate_var_schedule'   : [1,0],
}

run_sg(arg_dict, ppo, base_path="/data/acrobot/")

# model = ppoModel(
#     policy = MLP(input_size, output_size, num_layers, layer_size, activation),
#     value_fn = MLP(input_size, 1, num_layers, layer_size, activation),
#     action_var = 4
# )
#
# arg_dict = {
#     'env_name' : 'su_acrobot-v0',
#     'model' : model,
#     'num_epochs' : 500,
#     'action_var_schedule' : [1,0]
# }
# run_sg(arg_dict, ppo, base_path="/data/acrobot/")




arg_dict = {
    'env': 'su_acrobot-v0',
    'alg': 'ppo2',
    'network': 'mlp',
    'num_timesteps': '2e6',
    'num_env': '1',
    'num_layers': '3',
    'num_hidden': '64'
}

#run_baselines(arg_dict, run_name='discrete1', description='first run with discrete environment, appears to actually get positive rewards!')
#run_and_save_bs(arg_dict, run_name='/acrobot_baseline/', description='')

