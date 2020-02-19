import ray
from ray import tune
from ray.rllib.models import ModelCatalog
import json
import os
import time
import random
import string
import gym

from gym import envs

import pybullet_envs
import seagul.envs
from pathlib import Path
# import and register custom models
all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]
print(env_ids)
from rbf_net import RBFModel
from mlp_net import MLP, Linear
ModelCatalog.register_custom_model("RBF", RBFModel)
ModelCatalog.register_custom_model("MLP", MLP)
ModelCatalog.register_custom_model("linear", Linear)

#---- set parameters: --------------------
algos = {
    "high-throughput" : {
        0: "APEX",
        1: "IMPALA",
        2: "APPO"
    },
    "gradient-based" : {
        0: "A3C",
        1: "A2C",
        2: "DDPG",
        3: "PPO",
        4: "SAC",
        5: "TD3",
        6: "PG"
    },
    "derivative-free" : {
        0: "ARS",
        1: "ES"
    }
}
envs = {
    0: {"name": "HumanoidBulletEnv-v0", "stop": 6000},
    1: {"name": "Walker2DBulletEnv-v0", "stop": 2000},
    2: {"name": "Pendulum-v0", "stop": 150},
    3: {"name": "HalfCheetahBulletEnv-v0", "stop": 9000}}

ray.init()
for i in range(3,6):
    #---- adjust parameters: -------------------------------------
    algorithm = algos["gradient-based"][i]
    # algorithm = algos["0"]
    environment = envs[3]["name"]
    output_dir = "./data/" + environment + "/mlp_default/"
    if os.path.exists("./params/" + environment + "/" + algorithm + ".json"):
        config = json.load(open("./params/" + environment + "/" + algorithm + ".json"))
    else: # for cluster
        config = json.load(open("./seagul/seagul/rllib/rllib_with_rbf/params/" + environment + "/" + algorithm + ".json"))
    config['env'] = environment
    #---- tune hyperparameters: ----------------------------------
    config['Q_model'] = {'hidden_activation': 'relu',
                         'hidden_layer_sizes': [256, 256]}
    config['policy_model'] = {'hidden_activation': 'relu',
                              'hidden_layer_sizes': [256, 256]}
    # config['model'] = tune.grid_search([{"custom_model": "RBF", 
    #                                      "custom_options": {
    #                                          "normalization": False,
    #                                          "units": 64,
    #                                          "const_beta": False,
    #                                          "beta_initial": "ones"}},
    #                                     {"custom_model": "MLP",
    #                                      "custom_options": {
    #                                          "hidden_neurons": [64, 64]}},
    #                                     {"custom_model": "linear"}])
    # config['model'] = tune.grid_search([{"custom_model": "RBF", 
    #                                      "custom_options": {
    #                                          "normalization": False,
    #                                          "units": 64,
    #                                          "const_beta": False,
    #                                          "beta_initial": "ones"}},
    #                                     {"custom_model": "RBF", 
    #                                      "custom_options": {
    #                                          "normalization": True,
    #                                          "units": 64,
    #                                          "const_beta": True,
    #                                          "beta_initial": "ones"}},
    #                                     {"custom_model": "RBF", 
    #                                      "custom_options": {
    #                                          "normalization": True,
    #                                          "units": 64,
    #                                          "const_beta": False,
    #                                          "beta_initial": "ones"}},
    #                                     {"custom_model": "RBF", 
    #                                      "custom_options": {
    #                                          "normalization": False,
    #                                          "units": 64,
    #                                          "const_beta": True,
    #                                          "beta_initial": "ones"}}])
    #---------------------------------------------------------------
    try:
        analysis = tune.run(
            algorithm,
            local_dir=output_dir,
            # name="test",
            stop={"episode_reward_mean": [envs[x]["stop"] for x in envs if envs[x]["name"] == environment][0], "timesteps_total": 500000},
            checkpoint_freq=10,
            max_failures=5,
            checkpoint_at_end=True,
            config=config,
            num_samples=3
        )
    except Exception as e:
                Path(output_dir + algorithm).mkdir(parents=True, exist_ok=True)
                file = open(output_dir + algorithm  + "/" + "exception.txt", "w")
                file.write(str(e))
                file.close()

                # sbatch ./seagul/seagul/notebooks/pod/run_with_singularity.bash seagul/seagul/rllib/rllib_with_rbf/run.py -p short
