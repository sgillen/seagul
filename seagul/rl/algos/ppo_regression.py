import torch.nn as nn
from seagul.rl.algos.ppo import ppo
from seagul.nn import MLP, CategoricalMLP
import torch
import matplotlib.pyplot as plt
from seagul.rl.models import PpoModel

torch.set_default_dtype(torch.double)

%matplotlib inline

input_size = 3
output_size = 1
layer_size = 64
num_layers = 2
activation = nn.ReLU

policy = MLP(input_size, output_size, num_layers, layer_size, activation)
value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
model = PpoModel(policy, value_fn, action_var=.1, discrete=False)

# Define our hyper parameters
arg_dict = {
'epoch_batch_size' : 2048,  # how many steps we want to use before we update our gradients
'env_timesteps' : 199,  # number of steps in an episode (unless we terminate early)
'reward_stop' : -200,
'policy_batch_size' : 512,
'value_batch_size' : 512,
'v_epochs' : 10,
'p_epochs' : 10,
'policy_lr' : 1e-2,
'value_lr' : 1e-2,
'action_var_schedule' : [.707],
}

gamma = 0.99
lam = 0.99
eps = 0.2

t_model, rewards, var_dict = ppo("Pendulum-v0", 200, model, **arg_dict)
print(rewards)
plt.plot(rewards)

