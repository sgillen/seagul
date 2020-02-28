#%%
import numpy as np
from numpy import sin,cos,pi
import gym
import seagul.envs

from multiprocessing import Pool

from control import lqr, ctrb

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

#%%
m1=1; m2=1; l1=1; l2=2; lc1=.5; lc2=1; I1=.083; I2=.33; g=9.8;
max_torque = 25

th1 = pi/2; th2 = 0; th1d = 0; th2d = 0;

TAU = np.array([[0],[1]])

m11 = m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*cos(th2)) + I1 + I2;
m22 = m2*lc2**2 + I2;
m12 = m2*(lc2**2 + l1*lc2*cos(th2)) + I2;
M = np.array([[m11, m12], [m12, m22]]);

h1 = -m2*l1*lc2*sin(th2)*th2d**2 - 2*m2*l1*lc2*sin(th2)*th2d*th1d;
h2 = m2*l1*lc2*sin(th2)*th1d**2;
H = np.array([[h1],[h2]]);

phi1 = (m1*lc1+m2*l1)*g*cos(th1) + m2*lc2*g*cos(th1+th2);
phi2 = m2*lc2*g*cos(th1+th2);
PHI = np.array([[phi1], [phi2]])

Bl = np.linalg.inv(M)@TAU
Blin = np.array([[0, 0],[0, 0],[0, Bl[0].item()], [0, Bl[1].item()]])

DPHI = np.array([[-g*(m1*lc1 + m2*l1 + m2*lc2), -m2*lc2*g], [-m2*lc2*g, -m2*lc2*g]])
Al = -np.linalg.inv(M)@DPHI
Alin = np.array([[0,0, 1, 0],[0, 0, 0, 1], [Al[0,0], Al[0,1],0,0], [Al[1,0], Al[1,1], 0, 0]])

Ctr = ctrb(Alin, Blin)
assert np.linalg.matrix_rank(Ctr) == 4

Q = np.identity(4)
Q[0,0] = 5
Q[1,1] = 2
Q[2,2] = 1/2
Q[3,3] = 1/2

#Q = np.array([[1000, -500, 0,0],[-500, 1000, 0, 0],[0, 0, 1000, -500],[0,0,-500,1000]])

R = np.identity(2)*.01
K,S,E = lqr(Alin,Blin,Q,R)
k = np.array(K[1,:])
print(k)


def control(q):
    #import ipdb; ipdb.set_trace()
    gs = np.array([pi/2,0,0,0])
    #return 0
    return -k.dot(q - gs)

def reward_fn(s, a):
    reward = np.sin(s[0]) + 2 * np.sin(s[0] + s[1])
    done = reward < 0
    return reward, done

def do_rollout(args):
    x,trial_num = args
    th1, th2, dth1, dth2 = x
    np.random.seed(trial_num)

    obs = env.reset(init_vec = [th1,th2,dth1,dth2])

    local_state_hist = np.zeros((env.num_steps, env.observation_space.shape[0]))
    local_reward_hist = np.ones((env.num_steps, 1))*-1
    local_gate_hist = np.zeros((env.num_steps, 1))
    local_action_hist = np.zeros((env.num_steps, 1))

    for i in range(env.num_steps):
        actions = np.clip(np.asarray(control(obs)), -max_torque, max_torque)
        local_gate_hist[i] = 1
        obs, reward, done, _ = env.step(actions)
        local_action_hist[i, :] = np.copy(actions)
        local_state_hist[i, :] = np.copy(obs)
        local_reward_hist[i, :] = np.copy(reward)
        if done:
            break

    return local_action_hist, local_state_hist, local_reward_hist, local_gate_hist, i

#%%b

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
          "i2": I2,
          "reward_fn": reward_fn,
          "act_hold": 1
          }

env = gym.make('su_acrobot-v0', **config)

num_trials = 2000
action_hist = np.zeros((num_trials, env.num_steps, 1))
state_hist = np.zeros((num_trials, env.num_steps, env.observation_space.shape[0]))
reward_hist = np.zeros((num_trials, env.num_steps, 1))
gate_hist = np.zeros((num_trials, env.num_steps, 1))
good_list = []

obs = env.reset()

import time
start = time.time()

th1_min = pi/2-1; th1_max = pi/2+1
th2_min = -1; th2_max = 1
th1dot_min= -5; th1dot_max = 5
th2dot_min= -10; th2dot_max = 10

X = np.zeros((num_trials,4),dtype=np.float32)
Y = np.zeros((num_trials,1),dtype=np.float32)

samples = np.random.random_sample((num_trials,4))
samples*=np.array([th1_min - th1_max, th2_min - th2_max, th1dot_min - th1dot_max, th2dot_min - th2dot_max])
samples += np.array([th1_max, th2_max, th1dot_max, th2dot_max])
total_steps=0

pool = Pool()  # defaults to number of available CPU's
for ind, res in enumerate(pool.imap(do_rollout, zip(samples, range(num_trials)))):
    acts, obs, rews, gate,steps = res
    action_hist[ind, :, :] = acts
    state_hist[ind, :, :] = obs
    reward_hist[ind, :, :] = rews
    gate_hist[ind, :, :] = gate
    total_steps+=steps
    X[ind, :] = samples[ind,:]
    Y[ind] = sum(rews) > env.num_steps*3-10

Y *= np.sqrt(Y.shape[0]/sum(Y))/2

print(time.time() - start)


#%%
from seagul.nn import MLP, fit_model
import torch

net = MLP(4, 1, 2, 16) #output_activation=torch.nn.Softmax)
Y0 = np.ones((num_trials,1), dtype=np.float32)

class_weight = torch.tensor((Y.shape[0]/sum(Y))/2,  dtype=torch.float32)

loss_hist = fit_model(net, X, Y, 50, batch_size=2048, loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=class_weight))

#loss_hist = fit_model(net, X, Y, 100, batch_size=2048)
#loss_hist = fit_model(net, X, Y0, 5, batch_size=2048, loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=class_weight))

plt.close()
plt.plot(loss_hist)
plt.show()

#%%


n_thdot = 1
n_th = 1000

th1_vals = np.linspace(pi/2-1, pi/2+1, n_th)
th2_vals = np.linspace(-pi, pi, n_th)
th1dot = 0
th2dot = 0

#th1dot_vals = np.linspace(-10, 10, n_thdot)
#th2dot_vals = np.linspace(-15, 15, n_thdot)

th1dot_vals = th2dot_vals = [0];
sig = torch.nn.Sigmoid()


coords = np.zeros((n_th, n_th, 4), dtype=np.float32)

from itertools import product

start = time.time()
for i, j in product(range(n_th), range(n_th)):
    coords[i, j, :] = np.array([th1_vals[i], th2_vals[j], th1dot, th2dot])

preds = sig(net(coords.reshape(-1, 4))).reshape(n_th, n_th).detach()

end = time.time()
print(end - start)

fig, ax = plt.subplots(n_thdot, n_thdot, figsize=(22, 16))
# generate 2 2d grids for the x & y bounds
y, x = np.meshgrid(th1_vals, th2_vals)
z = preds

# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = z[:-1, :-1]
z_min, z_max = 0, np.abs(z).max()

c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
# ax.set_title('pcolormesh')
# set the limits of the plot to the limits of the data
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)
plt.show()

