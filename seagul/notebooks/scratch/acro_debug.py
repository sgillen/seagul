import seagul.envs
import gym

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import torch

from itertools import product
from multiprocessing import Pool
import time
from seagul.integration import rk4

import gym
import seagul.envs
from numpy import pi, sin, cos
import time
import matplotlib.pyplot as plt
from control import lqr, ctrb

from seagul.rl.models import PPOModelActHold
from seagul.nn import MLP
from seagul.rl.run_utils import load_workspace

from scipy.integrate import solve_ivp


state_hist = np.random.random((1,50,4))
index = 244
env = gym.make('su_acrobot-v0')
obs = env.reset()
for i in range(50):
    env.state = state_hist[0,i,:]
    env.render()
