import gym
import seagul.envs
import pickle
import torch.utils.data

from seagul.rl.run_utils import load_model, load_workspace
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd
from seagul.rl.models import PPOModel, SwitchedPPOModel, SwitchedPPOModelActHold
from seagul.nn import MLP

# torch.set_default_dtype(torch.double)
dtype = np.float32

import matplotlib.pyplot as plt

import seagul.envs
import numpy as np
from numpy import pi
import gym
from mpl_toolkits.mplot3d import Axes3D
# from simple_pid import PID
import gym

from torch.multiprocessing import Pool
from itertools import product
from seagul.plot import smooth_bounded_curve
import os

# %%

jup_dir = "/home/sgillen/work/"

def load_trials(trial_dir):
    directory = jup_dir + trial_dir

    ws_list = []
    model_list = []
    min_length = float('inf')
    for entry in os.scandir(directory):
        model, env, args, ws = load_workspace(entry.path)

        if len(ws["raw_rew_hist"]) < min_length:
            min_length = len(ws["raw_rew_hist"])

        ws_list.append(ws)
        model_list.append(model)

    min_length = int(min_length)
    rewards = np.zeros((min_length, len(ws_list)))
    for i, ws in enumerate(ws_list):
        rewards[:, i] = np.array(ws["raw_rew_hist"][:min_length])

    return ws_list, model_list, rewards


def do_rollout(init_point):
    env = gym.make(ws['env_name'], **ws['env_config'])
    if init_point is not None:
        obs = env.reset(init_point)
    else:
        obs = env.reset()

    action_hist = []
    m_act_hist = []
    obs_hist = []
    reward_hist = []
    done = False

    while not done:
        obs = torch.as_tensor(obs, dtype=torch.float32)

        actions, _, _, logp = model.step(obs.reshape(1, -1))
        actions = actions.detach()

        actions = torch.as_tensor(actions, dtype=torch.float32)
        obs, reward, done, _ = env.step(actions.detach().numpy())
        # env.render()
        # time.sleep(.1)

        obs_hist.append(obs)
        action_hist.append(actions)
        reward_hist.append(reward)

    obs_hist = np.stack(obs_hist)

    return obs_hist, action_hist, reward_hist, env.lqr_on


# %%

ws_list, model_list, _ = load_trials("seagul/seagul/notebooks/switching/data_needle/longer_thresh")
ws = ws_list[-1];
model = model_list[-1]

# init_state = np.array([0.34992436, 0.3857042 , 0.07708171, 0.59654361])
init_state = np.array([np.pi / 2, 0, 0, 0])
obs_hist, action_hist, reward_hist, lqr_on = do_rollout(np.array(init_state))

print("Sum of rewards: ", sum(reward_hist))
print("lqr on?: ", lqr_on)

env = gym.make(ws['env_name'], **ws['env_config'])
t = np.array([i * env.dt * env.act_hold for i in range(len(action_hist))])

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

ax[0].step(t, action_hist)
ax[1].plot(t, obs_hist, 'x--')
ax[1].legend(['th1', 'th2', 'th1dot', 'th2dot'])
plt.show()

# %% md

# Needle sac can be made to work well

## Observation: Bigger networks and longer runs improve performance (shocking...)

## worth noting the one successful rllib trial from last week was a [256, 256] network, and trying to replicate those results with a [32,32] failed

# %%

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

ws_list, model_list, rewards = load_trials("seagul/seagul/notebooks/switching/data_needle/long_small_strong")

smooth_bounded_curve(rewards, ax=ax[0])
ax[0].set_title('Hidden sizes: (32,32)')

ws_list, model_list, rewards = load_trials("seagul/seagul/notebooks/switching/data_needle/less_hack")
smooth_bounded_curve(rewards, ax=ax[1])
ax[1].set_title('Hidden sizes: (256,256)')

ws = ws_list[-1];
model = model_list[-1]

# %% md

# Again, reasonably robust to initial conditions

### This time environment is reset normally, but with randomized initial *position* (initial velocity is always zero, learning degrades dramatically with nonzero starting velocities)

# %%

# Generate "balance map" at slice dth = 0
pool = Pool()
th1_min = 0;
th1_max = 2 * pi;
num_th1 = 41
th1_vals = np.linspace(th1_min, th1_max, num_th1)

th2_min = -pi;
th2_max = pi;
num_th2 = 41
th2_vals = np.linspace(th2_min, th2_max, num_th2)

th_results = np.zeros((th1_vals.size, th2_vals.size))
th_lqr_results = np.zeros((th1_vals.size, th2_vals.size))
rewards = np.zeros((th1_vals.size, th2_vals.size))

all_obs_hist = []

end_point = np.array([1.57079633, 0., 0., 0.])

import time

start = time.time()

for i, res in enumerate(pool.imap(do_rollout, product(th1_vals, th2_vals, [0], [0]))):
    obs_hist, action_hist, reward_hist, lqr_on = res
    all_obs_hist.append(obs_hist)
    errs = np.sum(abs(obs_hist[-10:] - end_point), axis=1) < 1.5
    th_results.flat[i] = errs.all()
    th_lqr_results.flat[i] = lqr_on
    rewards.flat[i] = sum(reward_hist)

end = time.time()
print(end - start)

# %%

# Generate "balance map" at slice th = 0

dth1_min = -10;
dth1_max = 10;
num_dth1 = 41
dth1_vals = np.linspace(dth1_min, dth1_max, num_dth1)

dth2_min = -30;
dth2_max = 30;
num_dth2 = 41
dth2_vals = np.linspace(dth2_min, dth2_max, num_dth2)

dth_results = np.zeros((dth1_vals.size, dth2_vals.size))
dth_lqr_results = np.zeros((dth1_vals.size, dth2_vals.size))
rewards = np.zeros((dth1_vals.size, dth2_vals.size))

end_point = np.array([1.57079633, 0., 0., 0.])

import time

start = time.time()

for i, res in enumerate(pool.imap(do_rollout, product([0], [0], dth1_vals, dth2_vals))):
    obs_hist, action_hist, reward_hist, lqr_on = res
    errs = np.sum(abs(obs_hist[-5:] - end_point), axis=1) < 1.5
    dth_results.flat[i] = errs.all()
    dth_lqr_results.flat[i] = lqr_on
    rewards.flat[i] = sum(reward_hist)

end = time.time()
print(end - start)

# %%

fig, ax = plt.subplots(1, 2, figsize=(14, 7))

weird_list = []
for i in range(th1_vals.shape[0]):
    for j in range(th2_vals.shape[0]):
        if th_lqr_results[i, j] and th_results[i, j]:
            ax[0].plot(th1_vals[i], th2_vals[j], 'o', color='k', alpha=1)

        elif th_lqr_results[i, i] and not th_results[i, j]:
            ax[0].plot(th1_vals[i], th2_vals[j], 'o', color='r', alpha=1)

        elif th_results[i, j] and not th_lqr_results[i, j]:
            ax[0].plot(th1_vals[i], th2_vals[j], 'o', color='b', alpha=1)
            weird_list.append((i, j))

ax[0].set_title('256 network, balance map')
ax[0].set_xlabel('th1')
ax[0].set_ylabel('th2')

weird_list = []
for i in range(dth1_vals.shape[0]):
    for j in range(dth2_vals.shape[0]):
        if dth_lqr_results[i, j] and dth_results[i, j]:
            ax[1].plot(dth1_vals[i], dth2_vals[j], 'o', color='k', alpha=1)

        elif dth_lqr_results[i, i] and not dth_results[i, j]:
            ax[1].plot(dth1_vals[i], dth2_vals[j], 'o', color='r', alpha=1)

        elif dth_results[i, j] and not dth_lqr_results[i, j]:
            ax[1].plot(dth1_vals[i], dth2_vals[j], 'o', color='b', alpha=1)
            weird_list.append((i, j))

ax[1].set_title('256 network, balance map')
ax[1].set_xlabel('dth1')
ax[1].set_ylabel('dth2')
plt.show()

# %% md

# Manual curriculum learning

## starting with the policy trained above, train again but now with a stricter criteria for the goal state (must be in goal for 25 observations in the past)

# %%
#
# fig, ax = plt.subplots(1, 2, figsize=(16, 6))
#
# ws_list, model_list, rewards1 = load_trials("seagul/seagul/notebooks/switching/data_needle/long_strong_big")
#
# smooth_bounded_curve(rewards1, ax=ax[0])
# ax[0].set_ylim(-75, 100)
# ax[0].set_title('First round of training')

ws_list, model_list, rewards2 = load_trials("seagul/seagul/notebooks/switching/data_needle/less_hack")
smooth_bounded_curve(rewards2)
plt.ylim(-75, 100)
plt.title('Second round with stricter needle criteria')

ws = ws_list[-1];
model = model_list[-1]

# fig, axc = smooth_bounded_curve(np.concatenate((rewards1, rewards2)))
# axc.set_title('combined reward curve')
# plt.show()

# %% md

# Despite the reward curve, new policy does seem more robust

# %%

# Generate "balance map" at slice dth = 0
pool = Pool()
th1_min = 0;
th1_max = 2 * pi;
num_th1 = 41
th1_vals = np.linspace(th1_min, th1_max, num_th1)

th2_min = -pi;
th2_max = pi;
num_th2 = 41
th2_vals = np.linspace(th2_min, th2_max, num_th2)

err_list = []

th_results = np.zeros((th1_vals.size, th2_vals.size))
th_lqr_results = np.zeros((th1_vals.size, th2_vals.size))
rewards = np.zeros((th1_vals.size, th2_vals.size))

end_point = np.array([1.57079633, 0., 0., 0.])

import time

start = time.time()

for i, res in enumerate(pool.imap(do_rollout, product(th1_vals, th2_vals, [0], [0]))):
    obs_hist, action_hist, reward_hist, lqr_on = res
    errs = np.sum(abs(obs_hist[-10:] - end_point), axis=1) < .2
    #err_list.append(errs)
    th_results.flat[i] = errs.all()
    th_lqr_results.flat[i] = lqr_on
    rewards.flat[i] = sum(reward_hist)

end = time.time()
print(end - start)

# %%

# Generate "balance map" at slice th = 0

dth1_min = -10;
dth1_max = 10;
num_dth1 = 41
dth1_vals = np.linspace(dth1_min, dth1_max, num_dth1)

dth2_min = -30;
dth2_max = 30;
num_dth2 = 41
dth2_vals = np.linspace(dth2_min, dth2_max, num_dth2)

dth_results = np.zeros((dth1_vals.size, dth2_vals.size))
dth_lqr_results = np.zeros((dth1_vals.size, dth2_vals.size))
rewards = np.zeros((dth1_vals.size, dth2_vals.size))

end_point = np.array([1.57079633, 0., 0., 0.])

import time

start = time.time()

for i, res in enumerate(pool.imap(do_rollout, product([0], [0], dth1_vals, dth2_vals))):
    obs_hist, action_hist, reward_hist, lqr_on = res
    errs = np.sum(abs(obs_hist[-10:] - end_point), axis=1) < .2
    dth_results.flat[i] = errs.all()
    dth_lqr_results.flat[i] = lqr_on
    rewards.flat[i] = sum(reward_hist)

end = time.time()
print(end - start)

# %%

fig, ax = plt.subplots(1, 2, figsize=(14, 7))

weird_list = []
for i in range(th1_vals.shape[0]):
    for j in range(th2_vals.shape[0]):
        if th_lqr_results[i, j] and th_results[i, j]:
            ax[0].plot(th1_vals[i], th2_vals[j], 'o', color='k', alpha=1)

        elif th_lqr_results[i, i] and not th_results[i, j]:
            ax[0].plot(th1_vals[i], th2_vals[j], 'o', color='r', alpha=1)

        elif th_results[i, j] and not th_lqr_results[i, j]:
            ax[0].plot(th1_vals[i], th2_vals[j], 'o', color='b', alpha=1)
            weird_list.append((i, j))

ax[0].set_title('256 network, balance map')
ax[0].set_xlabel('th1')
ax[0].set_ylabel('th2')

weird_list = []
for i in range(dth1_vals.shape[0]):
    for j in range(dth2_vals.shape[0]):
        if dth_lqr_results[i, j] and dth_results[i, j]:
            ax[1].plot(dth1_vals[i], dth2_vals[j], 'o', color='k', alpha=1)

        elif dth_lqr_results[i, i] and not dth_results[i, j]:
            ax[1].plot(dth1_vals[i], dth2_vals[j], 'o', color='r', alpha=1)

        elif dth_results[i, j] and not dth_lqr_results[i, j]:
            ax[1].plot(dth1_vals[i], dth2_vals[j], 'o', color='b', alpha=1)
            weird_list.append((i, j))

ax[1].set_title('256 network, balance map')
ax[1].set_xlabel('dth1')
ax[1].set_ylabel('dth2')
plt.show()

# %% md

# %%

