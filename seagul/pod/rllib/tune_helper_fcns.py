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
