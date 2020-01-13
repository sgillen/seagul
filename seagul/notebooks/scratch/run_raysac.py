from seagul.rl.algos import sac, ppo
from seagul.rl.algos.sac_ray import ray_sac
from seagul.nn import MLP
from seagul.rl.models import SACModel, PPOModel

input_size = 17
output_size = 6
layer_size = 64
num_layers = 2

policy = MLP(input_size, output_size*2, num_layers, layer_size)
value_fn = MLP(input_size, 1, num_layers, layer_size)
q1_fn = MLP(input_size + output_size, 1, num_layers, layer_size)
q2_fn = MLP(input_size + output_size, 1, num_layers, layer_size)
model = SACModel(policy, value_fn, q1_fn, q2_fn, 3)


ppo_policy = MLP(input_size, output_size, num_layers, layer_size)
ppo_model = PPOModel(ppo_policy, value_fn)

env_name = "Walker2d-v2"
model, rews, var_dict = ray_sac(env_name, 100000, model, env_steps=1000, iters_per_update=100, min_steps_per_update=100, reward_stop=1000, exploration_steps=1000)
#model, rews, var_dict = ppo(env_name, 3e5, ppo_model)
globals().update(var_dict)