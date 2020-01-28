import gym
import seagul.envs

env_name = "su_acro_drake-v0"
env = gym.make(env_name)


from seagul.rl.run_utils import run_sg, run_and_save_bs
from seagul.rl.algos import ppo, ppo_switch
from seagul.rl.models import PPOModel, SwitchedPPOModel, SwitchedPPOModelActHold
from seagul.nn import MLP, CategoricalMLP, DummyNet


import torch
import torch.nn as nn


import numpy as np
from numpy import pi

from multiprocessing import Process

## init policy, valuefn
input_size = 4
output_size = 1
layer_size = 12
num_layers = 2
activation = nn.ReLU

#torch.set_default_dtype(torch.double)
proc_list = []

for seed in [0]:

    env_name = "su_acro_drake-v0"
    env = gym.make(env_name)

    # # hard coded gate for debugging
    # def gate(state):
    #     if len(state.shape) == 1:
    #         return (((140 * pi / 180 < state[0] < pi) and state[1] <= 0) or (
    #                 (pi < state[0] < 220 * pi / 180) and state[1] >= 0))
    #     else:
    #         ret  = ((((140 * pi / 180 < state[:,0]) & (state[:,0] < pi)) & (state[:,1] <= 0))
    #                | ((pi < state[:,0]) & (state[:,0] < 220 * pi / 180) & (state[:,1] >= 0)))
    #         return torch.as_tensor(ret,dtype=torch.double).reshape(-1,1)

    # hard coded gate for debugging
    # def gate(state):
    #     if len(state.shape) == 1:
    #         return ((140 * pi / 180 < state[0] < pi) or (pi < state[0] < 220 * pi / 180))
    #     else:
    #         ret  = ( ((140 * pi / 180 < state[:,0]) & (state[:,0] < pi)) | ((pi < state[:,0]) & (state[:,0] < 220 * pi / 180)))

    #         return torch.as_tensor(ret,dtype=torch.double).reshape(-1,1)

    #    gate_fn.net_fn = gate

    # def control(env,q):
    #     k = np.array([-1000, 1000, -10, -10])
    #     goal = np.copy(env.state)
    #     goal[0] -= pi
    #     return -k.dot(goal)

    def control(q):
        k = np.array([[1316.85000612, 555.41763935, 570.32667002, 272.57631536]],dtype=np.float32)
        # k = np.array([[278.44223126, 112.29125985, 119.72457377,  56.82824017]])
        gs = np.array([pi, 0, 0, 0],dtype=np.float32)
        # return 0
        return (-k.dot(gs - np.asarray(q)))

    model = SwitchedPPOModelActHold(
        # policy = MLP(input_size, output_size, num_layers, layer_size, activation),
        policy=torch.load("ppo2_warm_pol"),
        value_fn=torch.load("ppo2_warm_val"),
        # MLP(input_size, 1, num_layers, layer_size, activation),
        gate_fn=torch.load("gate_fn_ppo2"),
        nominal_policy=control,
        hold_count=200,
    )

    # model = SwitchedPPOModel(
    #     policy = torch.load("warm_policy_dr"),
    #     value_fn = torch.load("warm_value_dr"),
    #     gate_fn  = torch.load("gate_fn_dr"),
    #     # policy = MLP(input_size, output_size, num_layers, layer_size, activation),
    #     # value_fn = MLP(input_size, 1, num_layers, layer_size, activation),
    #     # gate_fn = CategoricalMLP(input_size, 1, num_layers, layer_size, activation),
    #     nominal_policy=control,
    #     env=None
    # )

    arg_dict = {
        "env_name": env_name,
        "model": model,
        "total_steps": 500*2048,
        "epoch_batch_size": 2048,
        "act_var_schedule": [2, 2],
        "gate_var_schedule": [0.1, 0.1],
        "gamma": 1,
        "seed": seed,
        "reward_stop" : 1500,
    }

    run_name = "1000_ppo2" + str(seed)

    #  import ipdb; ipdb.set_trace()
    run_sg(arg_dict, ppo_switch, run_name, 'trying to replicate earlier work that kinda of worked ', "/data/data1/switch4/")

#     p = Process(
#         target=run_sg,
#         args=(
#             arg_dict,
#             ppo_switch,
#             run_name,
#             "trying to replicate earlier results that use ppo with ppo2",
#             "/data/data2/drake_ppo2/",
#         ),
#     )
#     p.start()
#     proc_list.append(p)

# for p in proc_list:
#     print("joining")
#     p.join()


print("finished run ", run_name)
