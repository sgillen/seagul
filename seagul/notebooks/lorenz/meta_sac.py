import gym
import seagul.envs

from seagul.rl.run_utils import run_sg
from seagul.rl.algos import sac
from seagul.rl.models import SACModel
from seagul.nn import MLP

import torch.nn as nn
from multiprocessing import Process


env_name = "linear_z-v0"
#env_name = "lorenz-v0"
env = gym.make(env_name)

input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]
layer_size = 12
num_layers = 2
activation = nn.ReLU

for seed in [0,1,2,3]:

    policy = MLP(input_size, output_size * 2, num_layers, layer_size, activation)
    value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
    q1_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)
    q2_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)
    model = SACModel(policy=policy, value_fn=value_fn, q1_fn=q1_fn, q2_fn=q2_fn, act_limit=5)


    def reward_fn(s):
        if s[3] == 1:
            if s[0] > 2 and s[2] > 3:
                reward = 5.0
                s[3] = 0
            else:
                reward = -1.0

        elif s[3] == 0:
            if s[0] < -2 and s[2] < -3:
                reward = 5.0
                s[3] = 1
            else:
                reward = -1.0

        return reward, s


    env_config = {
        "num_steps": 500,
        "act_hold" : 10,
        "reward_fn": reward_fn,
        "xyz_max" : float('inf')
    }

    arg_dict = {
        "env_name": env_name,
        "model": model,
        "seed": seed,  # int((time.time() % 1)*1e8),
        "total_steps": 5e4,
        "gamma": 1,
        "env_config": env_config,
        "exploration_steps": 10000
    }

    run_name = "lorenz_me" + str(seed)

    proc_list = []
    p = Process(
        target=run_sg,
        args=(
            arg_dict,
            sac,
            run_name,
            "",
            "/data/seagul/sac",
        ),
    )
    p.start()
    proc_list.append(p)

for p in proc_list:
    p.join()


print("finished run ", run_name)
