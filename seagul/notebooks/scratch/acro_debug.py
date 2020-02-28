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

l1 = 1; l2 = 2
m1 = 1; m2 = 1
I2 = 1/12*m1*l1**2; I1 = 1/12*m2*l2**2
lc1 = l1/2; lc2 = l2/2
g = 9.8
th1 = -pi/2; th2 = 0
th1d = 0; th2d = 0


B = np.array([[0],[1]])
M = np.array([[I1 + I2 + m2*l1**2 + 2*m2*l1*lc2*cos(th2), I2 + m2*l1*lc2*cos(th2)], [I2 + m2*l1*lc2*cos(th2), I2]])
C = np.array([[-2*m2*l1*lc2*sin(th2)*th2d, -m2*l1*lc2*sin(th2)*th2d], [m2*l1*lc2*sin(th2)*th1d, 0]])
G = np.array([[-m1*g*lc1*cos(th1) - m2*g*(l1*cos(th1) + lc2*cos(th1 + th2))],[-m2*g*lc2*cos(th1+ th2)]])

Alin= np.array([[0,0, 1, 0],[0, 0, 0, 1], [g*(m1*l1 + m2*l1 + m2*lc2), (m2*g*lc2), 0, 0 ],[m2*g*lc2, m2*g*lc2, 0, 0]])
Bl = np.linalg.inv(M)*B
Blin = np.array([[0,0],[0,0],Bl[0,:], Bl[1,:]])

C = ctrb(Alin, Blin)
assert np.linalg.matrix_rank(C) == 4

Q = np.identity(4)
Q[0,0] = 5
Q[1,1] = 1
Q[2,2] = 1/2
Q[3,3] = 1/2

R = np.identity(2)*1
K,S,E = lqr(Alin,Blin,Q,R)
k = np.array(-K[1,:])
print(k)

max_torque = 25

config = {"init_state": [0, 0, 0, 0],
          "max_torque": max_torque,
          "init_state_weights": [0, 0, 0, 0],
          "dt": .01,
          "m2": m2,
          "m1": m1,
          "l1": l1,
          "lc1": lc1,
          "lc2": lc2,
          "i1": I1,
          "i2": I2
          }

env = gym.make('su_acrobot-v0', **config)

env.num_steps = 500


def control(q):
    # k = np.array([[278.44223126, 112.29125985, 119.72457377,  56.82824017]])
    # k = np.array([83.00702476, 27.2860448,  15.03836564,  3.12866625])
    # k = np.array([[100.48797281,  31.27961142,  18.25064425,   3.33078585]])
    # k =  np.array([1316.85, 555.42, 570.33, 272.58])
    # k = np.array([[15.35216187,  4.15795716,  8.98067486,  0.9426521 ]])
    # k = np.array([[273.77508024,  60.79140205, 159.03558811,   9.1835837 ]])
    # k = np.array([[209.05320712,  39.21744438, 120.78197876,   2.97587759]])
    # k = np.array([[266.08724059, 107.34926358, 114.95036444,  54.31747445]])
    # k = np.array([[269.552, 67.522, 98.966, 29.057]])
    # k = np.array([[20968.62606848,  3315.72482626,  3671.02961993,    32.92386574]])
    # k =  np.array([1316.85, 555.42, 570.33, 272.58])
    # k = np.array([[159.39046314,  39.98381558,  44.74660057,   2.77119863]])
    # k = np.array([204.06707408,  59.21086694,  37.41566423,   5.29518038])

    # k = np.array([[10.44, 3.561, 2.778, 1.301]])
    # k =
    # import ipdb; ipdb.set_trace()
    gs = np.array([-pi/2, 0, 0, 0])
    # return 0
    return -k.dot(gs - q)


action_hist = np.zeros((env.num_steps, 1))
action2_hist = np.zeros((env.num_steps, 1))

state_hist = np.zeros((env.num_steps, env.observation_space.shape[0]))
reward_hist = np.zeros((env.num_steps, 1))
obs = env.reset(init_vec=[-pi/2, .1, 0, 0])

import time

start = time.time()
for i in range(env.num_steps):
    actions = np.clip(np.asarray(control(obs)), -max_torque, max_torque)
    actions = np.array([0])
    # env.render()
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