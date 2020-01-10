import torch.nn as nn
from seagul.rl.algos.ppo2 import ppo
from seagul.nn import MLP, CategoricalMLP
import torch
import gym

import matplotlib.pyplot as plt

from seagul.rl.models import PpoModel


torch.set_default_dtype(torch.double)

import seagul.envs
import pybullet_envs

env_name = "Pendulum-v0"
env = gym.make(env_name)

input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]
layer_size = 64
num_layers = 3
activation = nn.ReLU

policy = MLP(input_size, output_size, num_layers, layer_size, activation)
value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
model = PpoModel(policy, value_fn, action_var=0.1, discrete=False)

# Define our hyper parameters
num_epochs = 100
batch_size = 2048  # how many steps we want to use before we update our gradients
num_steps = 1000  # number of steps in an episode (unless we terminate early)
max_reward = num_steps
p_batch_size = 1024
v_epochs = 1
p_epochs = 10
p_lr = 1e-2
v_lr = 1e-2

gamma = 0.99
lam = 0.99
eps = 0.2

# env2, t_policy, t_val, rewards = ppo('InvertedPendulum-v2', 100, policy, value_fn)
t_model, rewards, var_dict = ppo(env_name, 1e5, model, act_var_schedule=[1])
print(rewards[-1])
# print(rewards)
plt.plot(rewards)
plt.show()


# ===========================================
torch.set_default_dtype(torch.float32)

env_name = "Walker2DBulletEnv-v0"
env = gym.make(env_name)

input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]
layer_size = 64
num_layers = 3
activation = nn.ReLU

policy = MLP(input_size, output_size, num_layers, layer_size, activation)
value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
model = PpoModel(policy, value_fn, action_var=0.1, discrete=False)

# Define our hyper parameters
num_epochs = 100
batch_size = 2048  # how many steps we want to use before we update our gradients
num_steps = 1000  # number of steps in an episode (unless we terminate early)
max_reward = num_steps
p_batch_size = 1024
v_epochs = 1
p_epochs = 10
p_lr = 1e-2
v_lr = 1e-2

gamma = 0.99
lam = 0.99
eps = 0.2

# env2, t_policy, t_val, rewards = ppo('InvertedPendulum-v2', 100, policy, value_fn)
t_model, rewards, var_dict = ppo(env_name, 1e5, model, act_var_schedule=[1])
# print(rewards)
plt.plot(rewards)
plt.show()
