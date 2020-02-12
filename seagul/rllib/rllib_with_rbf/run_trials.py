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

def trial_str_creator(trial, algorithm, environment, model):
    trialname = algorithm + "_" + environment + "_" + model 
    # info_to_file(trialname + "_" + time.strftime("%Y-%m-%d_%H-%M-%S") + trial.trial_id)
    return trialname

def run_tune(algorithm, config, output_dir, comments):
    # ray.init()
    analysis = tune.run(
        algorithm,
        local_dir=output_dir,
        # name="test",
        # trial_name_creator=trial_str_creator,
        # stop={"episode_reward_mean": stop[environment]},
        stop={"timesteps_total": 500000},
        checkpoint_freq=1,
        max_failures=5,
        checkpoint_at_end=True,
        config=config,
        # trial_name_creator=trial_str_creator(algorithm, config['env'], config['model']['custom_model'])
    )
    # write comments to file
    if not (not comments): # if there are comments write to file
        trial_names = [str(trial) for trial in analysis.trials]
        for name in trial_names:
            dir_name = [dI for dI in os.listdir(output_dir + "/" + algorithm  + "/") if (os.path.isdir(os.path.join(output_dir + algorithm + "/",dI)) & dI.startswith(name))]
            file = open(output_dir + "/" + algorithm  + "/" + dir_name[0] + "/" + "info.txt", "w")
            file.write(comments)
            file.close()
    # ray.shutdown()