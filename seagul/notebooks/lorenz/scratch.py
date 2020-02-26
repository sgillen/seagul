import seagul.envs
import gym

env = gym.make('su_acrobot-v0')

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import torch

from itertools import product
from multiprocessing import Pool
import time

max_torque = 10

config = {"init_state": [-pi/8, pi/8, 0, 0], "max_torque": max_torque,
          "init_state_weights": [0, 0, 0, 0], "dt": .01
          }

env = gym.make('su_acrobot-v0', **config)

env.num_steps = 500


action_hist = np.zeros((env.num_steps, 1))
action2_hist = np.zeros((env.num_steps, 1))

state_hist = np.zeros((env.num_steps, env.observation_space.shape[0]))
reward_hist = np.zeros((env.num_steps, 1))
obs = env.reset()

import time

start = time.time()
for i in range(env.num_steps):
    actions = np.array([0])
    env.render()
    # obs = torch.as_tensor(obs, dtype=torch.float32)
    # actions = np.clip(np.asarray(pol(obs).detach()),-max_torque, max_torque)

    obs, reward, done, _ = env.step(actions)
    action_hist[i, :] = np.copy(actions)
    state_hist[i, :] = np.copy(obs)
    reward_hist[i, :] = np.copy(reward)
# if done:
#     break

print("total", time.time() - start)

plt.plot(action_hist)
plt.title('action hist')

plt.figure()
plt.plot(reward_hist)
plt.title('reward hist')

plt.figure()
plt.plot(state_hist)
plt.legend(['th1', 'th2', 'th3', 'th4'])

print(sum(reward_hist))
