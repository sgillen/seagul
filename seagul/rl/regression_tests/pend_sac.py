import gym
import torch.nn as nn
from seagul.rl.algos.sac import sac
from seagul.nn import MLP, CategoricalMLP
import torch
import matplotlib.pyplot as plt
from seagul.rl.models import SACModel
from multiprocessing import Process


# TODO for this to work need to convert from numpy dtype to torch
env_name = "Pendulum-v0"
# env = gym.make(env_name)
# dtype = env.action_space.sample().dtype


torch.set_default_dtype(torch.float32)



input_size = 3
output_size = 1
layer_size = 64
num_layers = 2
activation = nn.ReLU

policy = MLP(input_size, output_size*2, num_layers, layer_size, activation)

# Do I need to do weight sharing here?
value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
q_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)

model = SACModel(policy, value_fn, q_fn, 5)

def run_and_test(arg_dict,seed):
    t_model, rewards, var_dict = ppo("Pendulum-v0", 200*2048, model, seed=seed, **arg_dict)

    if (var_dict['early_stop']):
        print("seed", seed, "achieved 200 reward in ", len(rewards), "steps")
    #        print("Rewards were", rewards)

    else:
        print("Error: seed:" , seed, "failed")
        print("Rewards were", rewards)

    return
    

# Define our hyper parameters
arg_dict = {
    'epoch_batch_size' : 2048,  # how many steps we want to use before we update our gradients
    'sample_batch_size' : 2048,
    'reward_stop' : -200,
    'pol_batch_size' : 512,
    'val_batch_size' : 512,
    'val_epochs' : 10,
    'pol_epochs' : 10,
    'pol_lr' : 1e-2,
    'val_lr' : 1e-2,
}



sac(env_name, 1e4, model,**arg_dict)

# proc_list = []
# for seed in [0,1,2,3]:
#     p = Process(target=run_and_test, args=(arg_dict, seed))
#     p.start()
#     proc_list.append(p)


# for p in proc_list:
#     print("joining")
#     p.join()
