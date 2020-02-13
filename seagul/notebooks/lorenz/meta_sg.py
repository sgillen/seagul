import gym
import seagul.envs

from seagul.rl.run_utils import run_sg
from seagul.rl.algos import ppo
from seagul.rl.models import PPOModel, PPOModelActHold
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



    policy = MLP(input_size, output_size, num_layers, layer_size, activation)
    value_fn = MLP(input_size, 1, num_layers, layer_size, activation)

    # model = PPOModelActHold(
    #     policy=policy,
    #     value_fn=MLP(input_size, 1, num_layers, layer_size, activation),
    #     discrete=False,
    #     hold_count = 10
    # )

    model = PPOModel(policy=policy,value_fn=value_fn , discrete=False)


    def reward_fn(s):
        if s[3] == 1:
            if s[0] > 2 and s[2] > 3:
                reward = 5.0
                s[3] = 0
            else:
                reward = 0.0

        elif s[3] == 0:
            if s[0] < -2 and s[2] < -3:
                reward = 5.0
                s[3] = 1
            else:
                reward = 0.0

        return reward, s


    env_config = {
        "num_steps": 50,
        "reward_fn": reward_fn,
        "hold_count": 10
    }

    arg_dict = {
        "env_name": env_name,
        "model": model,
        "act_var_schedule": [1],
        "seed": seed,  # int((time.time() % 1)*1e8),
        "total_steps": 5e5,
        "epoch_batch_size": 2048,
        "gamma": 1,
        "pol_epochs": 10,
        "val_epochs": 10,
        "env_config": env_config
    }

    run_name = "ppo_again" + str(seed)

    proc_list = []
    p = Process(
        target=run_sg,
        args=(
            arg_dict,
            ppo,
            run_name,
            "",
            "/data/seagul/moar_ppo",
        ),
    )
    p.start()
    proc_list.append(p)

for p in proc_list:
    p.join()


print("finished run ", run_name)
