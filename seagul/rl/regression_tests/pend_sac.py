import torch.nn as nn
from seagul.rl.algos.sac import sac
<<<<<<< HEAD
from seagul.nn import MLP, RBF
import torch
from seagul.rl.models import SACModel
from multiprocessing import Process
import time
start_time = time.time()
=======
from seagul.nn import MLP
from seagul.rl.models import SACModel
from multiprocessing import Process, Manager
import sys


"""
Basic smoke test for SAC. This file contains an arg_dict that contains hyper parameters known to work with 
seagul's implementation of SAC. You can also run this script directly, which will check if the algorithm 
suceeds across 4 random seeds

Example:

from seagul.rl.regression_tests.pend_sac import arg_dict
from seagul.rl.algos import sac 
t_model, rewards, var_dict = sac(**arg_dict)  # Should get to -200 reward

"""
>>>>>>> 2edbab6eef86e092b47ad4167faaa07f3369a83f

input_size = 3
output_size = 1
layer_size = 64
num_layers = 2
activation = nn.ReLU

# policy = MLP(input_size, output_size * 2, num_layers, layer_size, activation)
# value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
# q1_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)
# q2_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)

policy = RBF(input_size, output_size * 2, layer_size)
value_fn = RBF(input_size, 1, layer_size)
q1_fn = RBF(input_size + output_size, 1, layer_size)
q2_fn = RBF(input_size + output_size, 1, layer_size)

model = SACModel(policy, value_fn, q1_fn, q2_fn, 1)

def run_and_test(arg_dict, retval):

    t_model, rewards, var_dict = sac(**arg_dict)

    seed = arg_dict["seed"]
    if var_dict["early_stop"]:
        print("seed", seed, "achieved 200 reward in ", len(rewards), "steps")
        retval[seed] = True
    else:
        print("Error: seed:", seed, "failed")
        print("Rewards were", rewards)
        retval[seed] = False


# Define our hyper parameters
arg_dict = {
    "env_name" : "Pendulum-v0",
    "model" : model,
    "total_steps" : 20000,
    "reward_stop" : -200
    }

if __name__ == "__main__" :
    #orig = sys.stdout
    #sys.stdout = open("/dev/null")

#    proc_list = []
#    manager = Manager()
#    ret_dict = manager.dict()
    ret_dict = {}
    for seed in [0,1,2,3]:
        arg_dict["seed"] = seed
        run_and_test(arg_dict,ret_dict)
#        p = Process(target=run_and_test, args=(arg_dict,ret_dict))
#        p.start()
#        proc_list.append(p)

    # for p in proc_list:
    #     print("joining")
    #     p.join()


    #sys.stdout = orig
    print(ret_dict)
