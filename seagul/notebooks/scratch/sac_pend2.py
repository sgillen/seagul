from seagul.rl.algos import sac, ppo
from seagul.rl.algos.sac_ray import ray_sac
from seagul.rl.algos.sac_sym import sac_sym
from seagul.nn import MLP
from seagul.rl.models import SACModel, PPOModel
import ray

input_size = 3
output_size = 1
layer_size = 64
num_layers = 2

ray.init()

policy = MLP(input_size, output_size*2, num_layers, layer_size)
value_fn = MLP(input_size, 1, num_layers, layer_size)
q1_fn = MLP(input_size + output_size, 1, num_layers, layer_size)
q2_fn = MLP(input_size + output_size, 1, num_layers, layer_size)
model = SACModel(policy, value_fn, q1_fn, q2_fn, 3)

ppo_policy = MLP(input_size, output_size, num_layers, layer_size)
ppo_model = PPOModel(ppo_policy, value_fn)

env_name = "Pendulum-v0"
#model, rews, var_dict = ray_sac(env_name, 20000, model, env_steps=0, iters_per_update=100, min_steps_per_update=100, reward_stop=-200, exploration_steps=100)
#model, rews, var_dict = ppo(env_name, 3e5, ppo_model)

model, rews, var_dict = sac(env_name, 160000, model, seed=0, env_steps=0, iters_per_update=100, min_steps_per_update=100, reward_stop=-200, exploration_steps=100)

#for seed in [0]:
#    %time model, rews, var_dict = sac(env_name, 20000, model, seed=seed, env_steps=0, iters_per_update=100, min_steps_per_update=100, reward_stop=-200, exploration_steps=100)


#globals().update(var_dict