from stable_baselines import PPO2, TRPO, DDPG, A2C
import seagul.envs
import gym

env_name = 'su_acrocot-v0'

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

import numpy as np
import matplotlib.pyplot as plt

import os
import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import DDPG
from stable_baselines.ddpg import AdaptiveParamNoiseSpec

best_mean_reward, n_steps = -np.inf, 0
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

models = []
model = PPO2('MlpPolicy', env,
             #nb_rollout_steps=500,
             #normalize_observations=True,
             #batch_size = 512,
             verbose=False,
            )

model.learn(100, seed=1)
