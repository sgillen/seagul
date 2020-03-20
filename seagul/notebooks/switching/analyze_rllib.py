import pandas as pd
import matplotlib.pyplot as plt
import json
import ray
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.ddpg as ddpg
import ray.rllib.agents.ddpg.td3 as td3

import ray.rllib.agents.sac as sac
import seagul.envs
import numpy as np
from numpy import pi
import gym
from mpl_toolkits.mplot3d import Axes3D
#from simple_pid import PID

import dill
import pickle
import gym
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from seagul.plot import smooth_bounded_curve
import os


#%% =====================================================================
directory = "/home/sgillen/work/seagul/seagul/notebooks/switching/data6/tune/sac/params_sweep2_10/SAC"
df_list = []
max_size = 0

for i, entry in enumerate(os.scandir(directory)):
    try:
        df = pd.read_csv(entry.path + "/progress.csv")
        df_list.append(df)

        if df.shape[0] > max_size:
            max_size = df.shape[0]

    except FileNotFoundError:
        pass

rewards = np.zeros((max_size, len(df_list)))

plt.figure()
for i, df in enumerate(df_list):
    plt.plot(df['episode_reward_mean'], 'ko--')
    trial_length = df['episode_reward_mean'].shape[0]
    rewards[:trial_length, i] = df['episode_reward_mean']

plt.show()
plt.figure()
fig, ax = smooth_bounded_curve(rewards, window=10)
ax.set_title('Smoothed reward curve, all seeds')
plt.show()

#%% =====================================================================
checkpoint_path = "/home/sgillen/work/seagul/seagul/notebooks/switching/data6/tune/sac/params_sweep2_10/SAC/SAC_su_acroswitch-v0_09c96804_2020-03-04_12-02-12d4e7vz9t/checkpoint_2499/checkpoint-2499"


config_path =  '/'.join(checkpoint_path.split('/')[:-2]) + '/params.pkl'
config = dill.load(open(config_path, 'rb'))
env_name = config['env']

csv_path = '/'.join(checkpoint_path.split('/')[:-2]) + '/progress.csv'
df = pd.read_csv(csv_path)
plt.plot(df['episode_reward_mean'], 'b')
plt.show()

ray.shutdown()
ray.init()

#trainer = ppo.PPOTrainer(config)
#trainer = td3.TD3Trainer(config)
#trainer = ddpg.DDPGTrainer(config)
trainer = sac.SACTrainer(config)

trainer.restore(checkpoint_path)

#%% =====================================================================
def do_rollout(init_point, render = False):
    env = gym.make(env_name, **config['env_config'])
    obs = env.reset(init_point)
    env.max_t = 20

    action_hist = []
    m_act_hist = []
    obs_hist = []
    reward_hist = []

    done = False

    while not done:
        actions, _, out_dict = trainer.compute_action(obs, full_fetch=True)
        obs, reward, done, _ = env.step(np.asarray(actions))

        if render:
            env.render()

        action_hist.append(np.copy(actions))
        obs_hist.append(np.copy(obs))
        reward_hist.append(np.copy(reward))

    action_hist = np.stack(action_hist)
    obs_hist = np.stack(obs_hist)
    # reward_hist = np.stack(reward_hist)

    return obs_hist, action_hist, reward_hist


def do_det_rollout(init_point):
    env = gym.make(env_name, **config['env_config'])
    obs = env.reset(init_point)
    action_hist = []
    m_act_hist = []
    obs_hist = []
    reward_hist = []
    env.max_t = 50

    done = False

    while not done:
        sampled_actions, _, out_dict = trainer.compute_action(obs, full_fetch=True)
        actions = out_dict['behaviour_logits'][0]

        obs, reward, done, _ = env.step(np.asarray(actions))
        env.render()
        action_hist.append(np.copy(actions))
        obs_hist.append(np.copy(obs))
        reward_hist.append(np.copy(reward))

    action_hist = np.stack(action_hist)
    obs_hist = np.stack(obs_hist)
    reward_hist = np.stack(reward_hist)

    return obs_hist, action_hist, reward_hist


#%% =====================================================================

def reward_fn(s, a):
    reward = -.1 * (np.sqrt((s[0] - pi / 2) ** 2 + .25 * s[1] ** 2))
    # reward = (np.sin(s[0]) + np.sin(s[0] + s[1]))
    return reward, False

obs_hist, action_hist, reward_hist = do_det_rollout(init_point=np.array([-pi / 2, 0, 0, 0]))
plt.plot(obs_hist, 'o--')
plt.title('Observations')
plt.legend(['th1', 'th1', 'th1d', 'th2d'])
plt.show()

print(sum(reward_hist))
plt.figure()
plt.plot(reward_hist, 'o--')
plt.title('reward_hist from trial')
plt.show()

fn_rews = []
for obs in obs_hist:
    r, _ = reward_fn(obs, 0)
    fn_rews.append(r)

print(sum(fn_rews))
plt.figure()
plt.plot(fn_rews, 'o--')
plt.title('reward_hist according to reference fn')
plt.show()

plt.figure()
plt.step([t for t in range(action_hist.shape[0])], action_hist)
plt.legend(['x', 'y'])
plt.title('Actions')
plt.show()