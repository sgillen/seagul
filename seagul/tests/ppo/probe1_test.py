import torch.nn as nn
from seagul.rl.ppo.ppo2 import PPOAgent
from seagul.nn import MLP
from seagul.rl.ppo.models import PPOModel
import seagul.envs

input_size = 1
output_size = 1
layer_size = 32
num_layers = 2
activation = nn.ReLU

policy = MLP(input_size, output_size, num_layers, layer_size, activation)
value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
model = PPOModel(policy, value_fn, init_logstd=-.5, learn_std=True)

# Define our hyper parameters
agent = PPOAgent(env_name="ProbeEnv1-v0", model=model, epoch_batch_size=2048, seed=0, sgd_batch_size=64,
                 lr_schedule=(1e-3,), sgd_epochs=30, target_kl=float('inf'), clip_val=True, env_no_term_steps=100,
                 normalize_return=True, normalize_obs=True, normalize_adv=True)

t_model, rewards, var_dict = agent.learn(total_steps = 1e6)