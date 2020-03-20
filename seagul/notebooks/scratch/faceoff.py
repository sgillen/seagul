from seagul.rl.models import PPOModel, SwitchedPPOModel, PPOModelActHold
from seagul.nn import MLP, CategoricalMLP
import numpy as np
import torch.nn as nn
import gym
import seagul.envs


env_name = "su_acrobot-v0"
max_torque = 5
max_t = 10

# init policy, valuefn
input_size = 4
output_size = 1
layer_size = 16
num_layers = 1
activation = nn.ReLU


def reward_fn(ns, act):
    reward = -1e-2 * (np.cos(ns[0]) + np.cos(ns[0] + ns[1]))

    done = False
    if abs(ns[0] - np.pi) < 1 and abs(ns[1]) < 1 and abs(ns[2]) < 3 and abs(ns[3]) < 3:
        reward += 2
        print("go zone")
        done = True

    return reward, done


env_config1 = {
    "init_state": [0, 0, 0, 0],
    "max_torque": max_torque,
    "init_state_weights": [0, 0, 0, 0],
    "dt": .01,
    "reward_fn" : reward_fn,
    "max_t" : max_t,
    "act_hold" : 20
}

policy = MLP(input_size, output_size, num_layers, layer_size, activation)
value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
model1 = PPOModel(
    policy=policy,
    value_fn=value_fn,
    action_var=1
)
env_config2 = {
    "init_state": [0, 0, 0, 0],
    "max_torque": max_torque,
    "init_state_weights": [0, 0, 0, 0],
    "dt": .01,
    "reward_fn" : reward_fn,
    "max_t" : max_t,
    "act_hold" : 1
}

model2 = PPOModelActHold(
    policy=policy,
    value_fn=value_fn,
    action_var=1,
    hold_count=20
)

inputs = np.random.random((128,1))
inputs = np.array(inputs, dtype=np.float32)

out1 = model1.policy(inputs)
out2 = model2.policy(inputs)

from seagul.rl.algos.ppo2 import do_rollout

env1 = gym.make(env_name, **env_config1)
env2 = gym.make(env_name, **env_config2)

ep_obs1, ep_act1, ep_rew1, ep_steps1 = do_rollout(env1, model1)
ep_obs2, ep_act2, ep_rew2, ep_steps2 = do_rollout(env2, model2)
