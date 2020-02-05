import ray
from ray import tune
from ray.rllib.models import ModelCatalog
import json
import gym
from gym import envs
import pybullet_envs
import os
import time
import random
import string
from pathlib import Path

# import and register custom models
from rbf_net import RBFModel1, RBFModel2
from mlp_net import MyKerasModel1, MyKerasModel2
ModelCatalog.register_custom_model("rbf", RBFModel1)
ModelCatalog.register_custom_model("rbf_2", RBFModel2)
ModelCatalog.register_custom_model("mlp_1_256", MyKerasModel1)
ModelCatalog.register_custom_model("mlp_2_64", MyKerasModel2)



#---- set parameters: --------------------
algorithm = "PPO" # "SAC"
environment = "HumanoidBulletEnv-v0"  # "Pendulum-v0"  #  "Walker2DBulletEnv-v0"   "HalfCheetahBulletEnv-v0" 
model = "mlp_2_64"
comments = "" # "mlp with two hidden layers with 64 neurons each"  # "rbf with new normalization code and different learning rates" # "rbf with old code (wrong for normalization) without input weights"
output_dir = "./data/"
#-----------------------------------------
config = json.load(open("./params/" + algorithm + "_" + environment + ".json"))
config['env'] = environment
config['model']['custom_model'] = model
#---- tune hyperparameters: --------------
# config['lr'] = tune.grid_search([0.01,0.001, 0.0003])
# config['lr'] = 0.01
#-----------------------------------------
# def info_to_file(name):
#     my_path = output_dir + algorithm  + "/" + name + "/" 
#     Path(my_path).mkdir(parents=True, exist_ok=True)
#     file = open(my_path + "info.txt", "w")
#     file.write(comments)
#     file.close()

def trial_str_creator(trial):
    trialname = algorithm + "_" + environment + "_" + model 
    # info_to_file(trialname + "_" + time.strftime("%Y-%m-%d_%H-%M-%S") + trial.trial_id)
    return trialname

ray.init()
analysis = tune.run(
    algorithm,
    local_dir=output_dir,
    # name="test",
    # trial_name_creator=trial_str_creator,
    stop={"episode_reward_mean": 6000},
    checkpoint_freq=1,
    max_failures=5,
    checkpoint_at_end=True,
    config=config,
    trial_name_creator=trial_str_creator
)

# write comments to file
if not (not comments): # if there are comments write to file
    trial_names = [str(trial) for trial in analysis.trials]
    for name in trial_names:
        dir_name = [dI for dI in os.listdir(output_dir + "/" + algorithm  + "/") if (os.path.isdir(os.path.join(output_dir + algorithm + "/",dI)) & dI.startswith(name))]
        file = open(output_dir + "/" + algorithm  + "/" + dir_name[0] + "/" + "info.txt", "w")
        file.write(comments)
        file.close()

