from multiprocessing import Process
import seagul.envs

import gym
env_name = "su_acrobot-v0"
env = gym.make(env_name)

import torch
import torch.nn as nn
import numpy as np
from numpy import pi

# init policy, value fn
input_size = 4
output_size = 1
layer_size = 32
num_layers = 2
activation = nn.ReLU

from seagul.rl.run_utils import run_sg, run_and_save_bs
from seagul.rl.sac import sac, SACModel, SACModelActHold
from seagul.nn import MLP, CategoricalMLP
from seagul.integration import euler, rk4

proc_list = []
trial_num = input("What trial is this?\n")

m1 = 1; m2 = 1
l1 = 1; l2 = 1
lc1 = .5; lc2 = .5
I1 = .2; I2 = 1.0
g = 9.8
max_torque = 25
max_t = 10

for seed in np.random.randint(0, 2 ** 32, 8):

    policy = MLP(input_size, output_size*2, num_layers, layer_size, activation)
    value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
    q1_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)
    q2_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)

    # model = SACModelActHold(
    #     policy=policy,
    #     value_fn = value_fn,
    #     q1_fn = q1_fn,
    #     q2_fn = q2_fn,
    #     act_limit = 25,
    #     hold_count = 20,
    # )

    model = SACModel(
        policy=policy,
        value_fn = value_fn,
        q1_fn = q1_fn,
        q2_fn = q2_fn,
        act_limit = max_torque,
    )

    def control(q):
        k = np.array([[-1649.86567367, -460.15780461, -716.07110032, -278.15312267]])
        gs = np.array([pi / 2, 0, 0, 0])
        return -k.dot(q - gs)

    def reward_fn_sin(s,a):
        reward = (np.sin(s[0]) + np.sin(s[0] + s[1]))
        return reward, False

    
    # def reward_fn(s, a):
    #     reward = -.1*(np.sqrt((s[0] - pi/2)**2 + s[1]**2))
    #     #reward = (np.sin(s[0]) + np.sin(s[0] + s[1]))
    #     return reward, False

    
    env_config = {
        "init_state": [-pi/2, 0, 0, 0],
        "max_torque": max_torque,
        "init_state_weights": [np.pi, np.pi, 0, 0],
        "dt": .01,
        "reward_fn" : reward_fn_sin,
        "max_t" : max_t,
        "m2": m2,
        "m1": m1,
        "l1": l1,
        "lc1": lc1,
        "lc2": lc2,
        "i1": I1,
        "i2": I2,
        "act_hold" : 20,
        "integrator": rk4,
    }

    alg_config = {
        "env_name": env_name,
        "model": model,
        "seed": seed,  # int((time.time() % 1)*1e8),
        "total_steps" : 2e6,
        "alpha" : .05,
        "exploration_steps" : 50000,
        "min_steps_per_update" : 500,
        "gamma": 1,
        "min_steps_per_update" : 500,
        "sgd_batch_size": 128,
        "replay_batch_size" : 4096,
        "iters_per_update": 4,
        #"iters_per_update": float('inf'),
        "env_config": env_config
    }

    p = Process(
        target=run_sg,
        args=(alg_config, sac, "sac-test", "no act hold this time", "/data_needle/" + trial_num + "/" + "seed" + str(seed)),
    )
    p.start()
    proc_list.append(p)


for p in proc_list:
    print("joining")
    p.join()
