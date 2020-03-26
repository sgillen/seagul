import gym
import numpy as np
import time
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


def evaluate(model, num_steps=1000):
  """
  Evaluate a RL agent
  :param model: (BaseRLModel object) the RL Agent
  :param num_steps: (int) number of timesteps to evaluate it
  :return: (float) Mean reward for the last 100 episodes
  """
  episode_rewards = []
  obs = env.reset()
  rew = 0.0
  for i in range(num_steps):
      # _states are only useful when using LSTM policies
      action, _states = model.predict(obs)

      obs, reward, done, info = env.step(action)
      
      # Stats
      rew = rew + reward
      if done:
          episode_rewards.append(rew)
          rew = 0.0
          obs = env.reset()
  # Compute mean reward for the last 100 episodes
  mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
  print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
  
  return mean_100ep_reward

def train(model, num_seeds=1, name = "mlp", num_ts = 2e4):
    rewards = []
    for seed in range(num_seeds):
        model_tmp = model
        name = name + time.strftime("%Y%m%d-%H%M") + "_" + str(seed)
        model_tmp.learn(total_timesteps=int(num_ts), tb_log_name=name, reset_num_timesteps=True)
        model_tmp.save("my_model_" + name)
        model_loaded = PPO2.load("my_model_" + name)
        mean_reward, std_reward = evaluate_policy(model_loaded, env, n_eval_episodes=10)
        rewards.append(mean_reward)
        del model_tmp
    [print("\n" + "mean reward: " + str(rew) + "\n") for rew in rewards]

def render(model, env):
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close() 

env = gym.make('HalfCheetah-v2')
policy_kwargs = dict(net_arch=[]) # act_fun=tf.nn.tanh
model = PPO2(MlpPolicy, env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./ppo_halfcheetah_tensorboard/")
# train(model=model, num_seeds=1, name = "linear", num_ts = 2e6)

# mean_reward = evaluate(model_loaded, num_steps=int(2e4)) # custom function
# render(PPO2.load("my_model_1"), env)

model_loaded = PPO2.load("my_model_linear20200310-1045_0")
mean_reward, std_reward = evaluate_policy(model_loaded, env, n_eval_episodes=10)
print("mean reward: " + str(mean_reward))