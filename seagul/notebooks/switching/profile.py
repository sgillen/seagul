import gym
import seagul.envs

env = gym.make('su_acro_drake-v0')

for i in range(100):
    env.reset()
