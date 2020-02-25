import pandas as pd
import os
import numpy as np
from seagul.plot import smooth_bounded_curve

jup_dir = "/home/sgillen/work/"
directory = jup_dir + "seagul/seagul/notebooks/lorenz/data/tune/euler_250_ah1/PPO"
df_list = []

for i, entry in enumerate(os.scandir(directory)):
    try:
        df_list.append(pd.read_csv(entry.path + "/progress.csv"))
    except FileNotFoundError:
        pass

rewards = np.zeros((df_list[0]['episode_reward_mean'].shape[0], len(df_list)))

for i, df in enumerate(df_list):
    rewards[:, i] = df['episode_reward_mean']

smooth_bounded_curve(rewards)