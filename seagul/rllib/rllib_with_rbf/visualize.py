import pandas as pd
import matplotlib.pyplot as plt
import json
import ray
import os
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.ppo.appo as appo
import ray.rllib.agents.ddpg as ddpg
import ray.rllib.agents.ddpg.td3 as td3
import ray.rllib.agents.sac as sac
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.a3c.a2c as a2c
import ray.rllib.agents.dqn.apex as apex
import ray.rllib.agents.impala as impala
import ray.rllib.agents.pg as pg
import ray.rllib.agents.ars as ars
import ray.rllib.agents.es as es
import seagul.envs
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
from numpy import pi
import gym
import pybullet_envs
from ray.rllib.models import ModelCatalog
from random import shuffle
import tensorflow as tf
import pickle
import re

from seagul.rllib.rllib_with_rbf.rbf_net import RBFModel
from seagul.rllib.rllib_with_rbf.mlp_net import MLP, Linear
def define_plot(colors, all_colors, output_dir, dir, smoothing_factor, cutoff):
    try:
        df = pd.read_csv(output_dir + dir + "/progress.csv")
    except Exception as e:
        print("Empty folder! \n" + str(e))
        return
    config = json.load(open(output_dir + dir + "/params.json"))
    try:
        model = config['model']['custom_model']
    except:
        model = "FCN"
    env = config["env"]
    alg = re.match('.+?(?=_)', os.path.basename(os.path.normpath(output_dir + dir))).group(0)
    if model == "RBF":
        try:
            if config['model']['custom_options']['normalization'] == False:
                model = model + "_no_normal"
            else:
                model = model + "_normal"
            if config['model']['custom_options']['const_beta'] == False:
                model = model + "_with_beta"
            else:
                model = model + "_const_beta"
        except:
            pass
    if model == "MLP":
        try:
            model = model + "_" + str(config['model']['custom_options']['hidden_neurons'])
            model = model + "_ts-per-iter-" + str(config['timesteps_per_iteration'])
        except:
            pass
    if model in colors:
        line_color = colors[model]
    else:
        colors[model] = all_colors[0]
        line_color = colors[model]
        all_colors.pop(0)
    try:
        cutoff_idx = df.index[df['timesteps_total'] == cutoff][0] if cutoff != -1 else -1
    except Exception as e:
        print("Redefine cutoff index. It might be too high. \n" + str(e))
        cutoff_idx = -1
    if smoothing_factor == 0:
        plt.plot(df['timesteps_total'][:cutoff_idx], df['episode_reward_mean'][:cutoff_idx], line_color, label = model)
    else:
        plt.plot(df['timesteps_total'][:cutoff_idx], gaussian_filter1d(df['episode_reward_mean'][:cutoff_idx], sigma=smoothing_factor), line_color, label = model)
    plt.title("Environment: " + env + ",  Algorithm: " + alg)

def plot_progress(output_dir, smoothing_factor = 0, cutoff = -1):
    colors = {'RBF': 'g', 'MLP': 'b', 'linear': 'c'}
    all_colors = ['m', 'y', 'k', 'r']
    # shuffle(all_colors)
    if os.path.exists(output_dir +  "/progress.csv"): # if already in folder and no looping required
        define_plot(colors, all_colors, output_dir, "", smoothing_factor, cutoff)
    else:
        for subdir, dirs, files in os.walk(output_dir):
            for dir in dirs:
                define_plot(colors, all_colors, subdir, dir, smoothing_factor, cutoff)
            break
    plt.xlabel('timesteps')
    plt.ylabel('episode reward mean')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

def render(checkpoint, home_path):
    alg = re.match('.+?(?=_)', os.path.basename(os.path.normpath(home_path))).group(0)
    current_env = re.search("(?<=_).*?(?=_)", os.path.basename(os.path.normpath(home_path))).group(0)
    checkpoint_path = home_path + "checkpoint_" + str(checkpoint) + "/checkpoint-" + str(checkpoint)
    config = json.load(open(home_path + "params.json"))
    config_bin = pickle.load(open(home_path + "params.pkl", "rb"))
    ray.shutdown()
    import pybullet_envs
    ray.init()
    ModelCatalog.register_custom_model("RBF", RBFModel)
    ModelCatalog.register_custom_model("MLP", MLP)
    ModelCatalog.register_custom_model("linear", Linear)

    if alg == "PPO":
        trainer = ppo.PPOTrainer(config)
    if alg == "SAC":
        trainer = sac.SACTrainer(config)
    if alg == "DDPG":
        trainer = ddpg.DDPGTrainer(config)
    if alg == "PG":
        trainer = pg.PGTrainer(config)
    if alg == "A3C":
        trainer = a3c.A3CTrainer(config)
    if alg == "TD3":
        trainer = td3.TD3Trainer(config)
#   "normalize_actions": true,
    trainer.restore(checkpoint_path)

    if "Bullet" in current_env:
        env = gym.make(current_env, render=True)
    else:
        env = gym.make(current_env)
    #env.unwrapped.reset_model = det_reset_model
    env._max_episode_steps = 10000
    obs = env.reset()

    action_hist = []
    m_act_hist = []
    state_hist  = []
    obs_hist = []
    reward_hist = []

    done = False
    for t in range(10000):
        # for some algorithms you can get the sample mean out, need to change the value on the index to match your env for now
        # mean_actions = out_dict['behaviour_logits'][:17]
        # actions = trainer.compute_action(obs.flatten())
        sampled_actions, _ , out_dict = trainer.compute_action(obs.flatten(),full_fetch=True)
        
        actions = sampled_actions
        
        obs, reward, done, _ = env.step(np.asarray(actions))
        
        env.render()
        
        action_hist.append(np.copy(actions))
        obs_hist.append(np.copy(obs))
        reward_hist.append(np.copy(reward))
    print(sum(reward_hist))
    print((obs_hist))
    #plt.plot(action_hist)
    #plt.figure()
    #plt.figure()
    #plt.plot(obs_hist)
    #plt.figure()

    # Reminder that the bahavior logits that come out are the mean and logstd (not log mean, despite the name logit)
    trainer.compute_action(obs, full_fetch=True)