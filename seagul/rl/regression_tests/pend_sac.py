import torch.nn as nn
from seagul.rl.sac.sac import sac
from seagul.nn import RBF
import torch
from seagul.rl.models import SACModel
import time
from multiprocessing import Process, Manager

"""
Basic smoke test for SAC. This file contains an arg_dict that contains hyper parameters known to work with 
seagul's implementation of SAC. You can also run this script directly, which will check if the algorithm 
suceeds across 4 random seeds

Example:

from seagul.rl.regression_tests.pend_sac import arg_dict
from seagul.rl.algos import sac 
t_model, rewards, var_dict = sac(**arg_dict)  # Should get to -200 reward

"""
start_time = time.time()

input_size = 3
output_size = 1
layer_size = 128
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
    torch.set_num_threads(1)

    t_model, rewards, var_dict = sac(**arg_dict)

    seed = arg_dict["seed"]
    if var_dict["early_stop"]:
        print("seed", seed, "achieved 200 reward in ", len(rewards), "steps")
        retval[seed] = True
    else:
        print("Error: seed:", seed, "failed")
        print("Rewards were", rewards[-1])
        retval[seed] = False


# Define our hyper parameters
arg_dict = {
    "env_name" : "Pendulum-v0",
    "model" : model,
    "total_steps" : 20000,
    "reward_stop" : -200,
    "use_gpu" : False,
    }

if __name__ == "__main__" :
    #orig = sys.stdout
    #sys.stdout = open("/dev/null")

    proc_list = []
    manager = Manager()
    ret_dict = manager.dict()

    for seed in [0,1,2,3]:
        arg_dict["seed"] = seed
        p = Process(target=run_and_test, args=(arg_dict,ret_dict))
        p.start()
        proc_list.append(p)

    for p in proc_list:
        p.join()


    #sys.stdout = orig
    print(ret_dict)
print("--- %s seconds ---" % (time.time() - start_time))