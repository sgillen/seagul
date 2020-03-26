import pandas as pd
import matplotlib.pyplot as plt
import json
import ray
import time
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
from scipy.interpolate import interp1d
import re
from seagul.plot import smooth_bounded_curve
import pybullet as p

from custom_models.rbf_net import RBFModel
from custom_models.mlp_net import MLP, Linear

# HELPER functions for analyzing rllib results (called in analyze.py)

def get_params(res_dir):
    """
    Returns the environment and algorithm of the first config file it finds.

    Example:
        from seagul.rllib.analysis_functions import get_params
        env, alg = get_params("./Results/SAC/")

        info: it works as well if the path is already the specific output folder like here -->
        env, alg = get_params("./Results/SAC/SAC_Walker2DBulletEnv-v0_4fac9f1e_2020-02-12_17-08-14wpyw1bxk/")

    Arguments:
         res_dir: string containing the directory to the output files

    Returns:
        env: string containig environment of the first config file it finds 
        alg: string containing algorithm of the first config file it finds
    """
    for output_dir in res_dir:
        if os.path.exists(output_dir + "params.json"): # if already in folder and no looping required
            try:
                config = json.load(open(output_dir + "params.json"))
                env = config["env"]
                alg = re.match('.+?(?=_)', os.path.basename(os.path.normpath(output_dir))).group(0)
                return env, alg
            except:
                pass
        else:
            for subdir, dirs, files in os.walk(output_dir):
                for dir in dirs:
                    try:
                        config = json.load(open(subdir + dir + "/params.json"))
                        env = config["env"]
                        alg = re.match('.+?(?=_)', os.path.basename(os.path.normpath(subdir + dir))).group(0)
                        return env, alg
                    except:
                        pass

def add(subdir, dir, cutoff, colors, all_colors):
    """
    subfuction used in outputs_to_df function
    """
    try:
        df = pd.read_csv(subdir + dir + "/progress.csv")
    except Exception as e:
        print("Empty folder! \n" + str(e))
        return
    config = json.load(open(subdir + dir + "/params.json"))
    try:
        model = config['model']['custom_model']
    except:
        model = "FCN"
        try:
            model = model + "_" + str(config['model']['fcnet_hiddens'])
        except:
            pass
    env = config["env"]
    alg = re.match('.+?(?=_)', os.path.basename(os.path.normpath(subdir + dir))).group(0)
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
        except:
            pass
    if model in colors:
        line_color = colors[model]
    else:
        colors[model] = all_colors[re.match('.+?(?=_|$)', model).group(0)][0]
        line_color = colors[model]
        all_colors[re.match('.+?(?=_|$)', model).group(0)].pop(0)
    try:
        cutoff_idx = df.index[df['timesteps_total'] == cutoff][0] if cutoff != -1 else -1
    except Exception as e:
        print("Redefine cutoff index. It might be too high. \n" + str(e))
        cutoff_idx = -1
    return model, df['timesteps_total'][:cutoff_idx], df['episode_reward_mean'][:cutoff_idx], line_color
    # return model, df['timesteps_total'][:cutoff_idx], df['episode_reward_max'][:cutoff_idx], line_color
    # return model, df['timesteps_total'][:cutoff_idx], df['time_total_s'][:cutoff_idx], line_color
    
def outputs_to_df(res_dir, cutoff = -1):
    """
    Returns a pandas dataframe containig all data found in the output directotry. 
    The dataframe can be used as an input for the plot_outputs function.
    the dataframe contains:
    - model: name of model used (MLP / linear / ...)
    - ts: time steps data
    - rewards: mean episode rewards
    - color: color for plot () - redish colors for RBF, greenish colors for MLP, yellow for linear and blueish for FCN

    Example:
        import seagul.rllib.analysis_functions
        entries = analysis_functions.outputs_to_df(res_dir="./Results/SAC/", cutoff=-1)
        analysis_functions.plot_outputs(entries)

        info: it works as well if the path is already the specific output folder like here -->
        res_dir = "./Results/SAC/SAC_Walker2DBulletEnv-v0_4fac9f1e_2020-02-12_17-08-14wpyw1bxk/"

    Arguments:
        res_dir: string containing the directory to the output files
        cutoff: int containing the maximal time step to be saved into the dataframe, if None specified all data is saved

    Returns:
        all_results: pd.DataFrame containig all data found in the output directotry
    """

    all_results = pd.DataFrame(columns = ['model', 'ts', 'rewards', 'color'])
    all_colors = {'RBF': ['#E85E10', '#BF2E0F', '#BE7F72', '#E6A092'], 'MLP': ['#9AE692', '#2BB51D', '#26FE11', '#578653'],'mlp': ['#9AE692', '#2BB51D', '#26FE11', '#578653'], 'linear': ['#DDEA11'], 'FCN': ['#1166EA', '#5B93E9', '#526F9C', '#5898FA']}
    colors = {}
    for output_dir in res_dir:
        if os.path.exists(output_dir +  "/progress.csv"): # if already in folder and no looping required
            model, ts, rewards, color = add(output_dir, "", cutoff, colors, all_colors)
            all_results.loc[len(all_results)] = [model, ts, rewards, color]
        else:
            for subdir, dirs, files in os.walk(output_dir):
                for dir in dirs:
                    model, ts, rewards, color = add(subdir, dir, cutoff, colors, all_colors)
                    all_results.loc[len(all_results)] = [model, ts, rewards, color]
                break
    return all_results

def plot_outputs(entries):
    """
    Takes a pandas dataframe as an input and uses the seagul.plot function to create a plot.
    """
    all_models = np.unique(np.array(entries['model']))
    ax = None
    for model in all_models:
        entries_of_model = entries.loc[entries['model'] == model]
        for i in range(len(entries_of_model['ts'])):
            cutoff = entries_of_model['ts'].iloc[i].shape[0] if 'cutoff' not in locals() or entries_of_model['ts'].iloc[i].shape[0] < cutoff else cutoff
            ts_min = int(entries_of_model['ts'].iloc[i].iloc[0]) if 'ts_min' not in locals() or int(entries_of_model['ts'].iloc[i].iloc[0]) > ts_min else ts_min
            ts_max = int(entries_of_model['ts'].iloc[i].iloc[-1]) if 'ts_max' not in locals() or int(entries_of_model['ts'].iloc[i].iloc[-1]) < ts_max else ts_max
        for i in range(len(entries_of_model['ts'])):
            rew_f = interp1d(entries_of_model['ts'].iloc[i].to_numpy(dtype=float), entries_of_model['rewards'].iloc[i].to_numpy(dtype=float))
            ts = np.linspace(ts_min, ts_max, num=1000)
            try:
                # rew = np.vstack((rew, entries_of_model['rewards'].iloc[i].to_numpy(dtype=float)[:cutoff]))
                rew = np.vstack((rew, rew_f(ts)))
            except:
                rew = rew_f(ts)
        # ts = entries_of_model['ts'].iloc[0].to_numpy(dtype=float)[:cutoff]
        col = entries_of_model['color'].iloc[0]
        where_is_nan = np.isnan(rew)
        rew[where_is_nan] = 0
        try:
            rew.shape[1]
        except:
            rew = np.expand_dims(rew,0)
        fig, ax = smooth_bounded_curve(data=np.transpose(rew), time_steps=ts, label=model, ax=ax, color = col, alpha=0.1)
        rew = 0

def render(checkpoint, home_path):
    """
    Renders pybullet and mujoco environments.
    """
    alg = re.match('.+?(?=_)', os.path.basename(os.path.normpath(home_path))).group(0)
    current_env = re.search("(?<=_).*?(?=_)", os.path.basename(os.path.normpath(home_path))).group(0)
    checkpoint_path = home_path + "checkpoint_" + str(checkpoint) + "/checkpoint-" + str(checkpoint)
    config = json.load(open(home_path + "params.json"))
    config_bin = pickle.load(open(home_path + "params.pkl", "rb"))
    ray.shutdown()
    import pybullet_envs
    ray.init()
    ModelCatalog.register_custom_model("RBF", RBFModel)
    ModelCatalog.register_custom_model("MLP_2_64", MLP)
    ModelCatalog.register_custom_model("linear", Linear)

    if alg == "PPO":
        trainer = ppo.PPOTrainer(config_bin)
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
    if alg == "ES":
        trainer = es.ESTrainer(config)
    if alg == "ARS":
        trainer = ars.ARSTrainer(config)
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
    step = 0

    for t in range(10000):
        # for some algorithms you can get the sample mean out, need to change the value on the index to match your env for now
        # mean_actions = out_dict['behaviour_logits'][:17]
        # actions = trainer.compute_action(obs.flatten())
        # sampled_actions, _ , out_dict = trainer.compute_action(obs.flatten(),full_fetch=True)
        sampled_actions = trainer.compute_action(obs.flatten())
        # sampled_actions, _ , out_dict = trainer.compute_action(obs.flatten(),full_fetch=True)
        
        actions = sampled_actions
        
        obs, reward, done, _ = env.step(np.asarray(actions))
        # env.camera_adjust()
        env.render(mode='human')
        time.sleep(0.01)
        # env.render()
        # env.render(mode='rgb_array', close = True)
        # p.computeViewMatrix(cameraEyePosition=[0,10,5], cameraTargetPosition=[0,0,0], cameraUpVector=[0,0,0])

        # if step % 1000 == 0:
        #     env.reset()
        # step += 1
        
        action_hist.append(np.copy(actions))
        obs_hist.append(np.copy(obs))
        reward_hist.append(np.copy(reward))
        if done:
            obs = env.reset()
    # print(sum(reward_hist))
    # print((obs_hist))
    #plt.plot(action_hist)
    #plt.figure()
    #plt.figure()
    #plt.plot(obs_hist)
    #plt.figure()

    # Reminder that the bahavior logits that come out are the mean and logstd (not log mean, despite the name logit)
    # trainer.compute_action(obs, full_fetch=True)
    trainer.compute_action(obs)
