#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
#torch.set_default_dtype(torch.double)
dtype = np.float32


# In[17]:


import os
jup_dir = "/home/sgillen/work/"
directory = jup_dir + "seagul/seagul/notebooks/switching/data5/acro2/trialgyst_v3.0t10/"
ws_list = []
model_list = []
max_size = 0
for entry in os.scandir(directory):
    model, env, args, ws = load_workspace(entry.path)
    plt.plot(ws["raw_rew_hist"])
    if len(ws["raw_rew_hist"]) > max_size:
        max_size = len(ws["raw_rew_hist"])

    plt.yscale
    plt.figure()
    print(entry.path)
    ws_list.append(ws)
    model_list.append(model)

plt.show()

# In[18]:
from seagul.plot import smooth_bounded_curve

rewards = np.zeros((max_size, len(ws_list)))

for i,ws in enumerate(ws_list):
    #plt.plot(ws["raw_rew_hist"])
    #plt.figure()
    #print(len(ws["raw_rew_hist"]))

    rewards[:len(ws["raw_rew_hist"]),i] = np.array(ws["raw_rew_hist"])
    
fig, ax = smooth_bounded_curve(rewards,window=100)
plt.show()
#ws = ws_list[0]
#model = model_list[0]


# In[19]:


plt.plot(ws['pol_loss_hist'])
plt.figure()
plt.plot(ws['val_loss_hist'])
plt.figure()
plt.plot(ws['raw_rew_hist'])
plt.show()

# In[ ]:


#torch.save(model.value_fn, open('./warm/ppo2_warm_valh2','wb'))
#torch.save(model.policy, open('./warm/ppo2_warm_polh2','wb'))
plt.show()

# In[3]:


#model.value_fn.state_dict()
ws=ws_list[0]


# In[22]:


env = gym.make(ws['env_name'], **ws['env_config'])
max_torque = ws['env_config']['max_torque']

action_hist = np.zeros((env.num_steps,1))
state_hist = np.zeros((env.num_steps, env.observation_space.shape[0]))
reward_hist = np.zeros((env.num_steps, 1))
logp_hist = np.zeros((env.num_steps, 1))
gate_mean = np.zeros((env.num_steps,1))
means_hist = np.zeros((env.num_steps,1))
obs = env.reset()
env.state = np.array([1.5707964,  0.       ,  0.       ,  0.], dtype=np.float32)

for i in range(env.num_steps):
    obs = torch.as_tensor(obs, dtype=torch.float32)
    actions, value, _, logp = model.step(obs)
    actions = np.clip(actions, -max_torque, max_torque)
    actions = torch.as_tensor(actions, dtype=torch.float32)
    #noise = torch.randn(1, 1)
    #actions, logp = model.select_action(obs, noise)

    obs, reward, done, _ = env.step(actions.detach().numpy())


    env.render()
    action_hist[i,:] = np.copy(actions.detach())
    state_hist[i,:] = np.copy(obs)
    reward_hist[i,:] = np.copy(reward)
    #logp_hist[i,:]   = np.copy(logp.detach())
    #means_hist[i, :] = np.copy(means.detach())

    if done:
        break

# for _ in range(1000):
#     obs = env.reset()
#     for i in range(env.num_steps):
#         obs = torch.as_tensor(obs, dtype=torch.float32).reshape(1,-1)
#         #actions, _, _, logp = model.step(obs)
#
#
#
#
#         obs, reward, done, _ = env.step(actions.detach().numpy())
#
#
#         #env.render()
#         action_hist[i,:] = np.copy(actions.detach())
#         state_hist[i,:] = np.copy(obs)
#         reward_hist[i,:] = np.copy(reward)
#         logp_hist[i,:]   = np.copy(logp.detach())
#         #means_hist[i, :] = np.copy(means.detach())
#
#         if done:
#             break


#gate_hist = [1 if p == 0.0 else 0 for p in logp_hist]


# In[15]:


t = np.array([i*env.dt*env.act_hold for i in range(action_hist.shape[0])])

plt.step(t, action_hist)
plt.figure()
plt.plot(t, state_hist)
plt.legend(['th1', 'th2', 'th1dot', 'th2dot'])
plt.show()

plt.figure()
plt.step(t, np.exp(logp_hist))
plt.show()

plt.figure()
plt.step(t, means_hist)
plt.show()


# In[10]:


print(sum(reward_hist))
plt.plot(reward_hist)
plt.show()
plt.figure()
plt.plot((np.sin(state_hist[:,0]) + np.sin(state_hist[:,0] + state_hist[:,1])))
plt.show()


# In[46]:


plt.plot(means_hist)
plt.figure()


