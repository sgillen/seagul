from baselines.common.cmd_util import make_mujoco_env
from baselines.common import tf_util as U
import tensorflow as tf
from baselines import logger
import os
from baselines.ppo1 import pposgd_simple

import policies.mlp_relu_policy as mlp_relu_policy

# Needed for saving
import datetime, time


def train(env_id, num_timesteps, seed=0):
    U.make_session(num_cpu=16).__enter__()

    def policy_fn(name, ob_space, ac_space):
        # return cartpole_policy.CartPolePolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=12, num_hid_layers=2)
        # return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=64, num_hid_layers=64)
        return mlp_relu_policy.ReluMlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=64,
                                             num_hid_layers=64)

    env = make_mujoco_env(env_id, seed)
    pi = pposgd_simple.learn(env, policy_fn,
                             max_timesteps=num_timesteps,
                             timesteps_per_actorbatch=2048,
                             clip_param=0.2, entcoeff=0.0,
                             optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                             gamma=0.99, lam=0.95, schedule='linear',
                             )
    env.close()

    return pi


def save_results(filename, description=None):
    """
    description: saves the results of a run of the second cell (the one that calls train) in this notebook

    """

    save_dir = "data/" + filename + "/"
    os.makedirs(save_dir)

    if description is None:
        description = input("please enter a description of the run")

    datetime_str = str(datetime.datetime.today())
    datetime_str = datetime_str.replace(" ", "_")

    runtime_str = str(datetime.timedelta(seconds=runtime))

    readme = open(save_dir + "README.txt", "w+")
    readme.write("datetime: " + datetime_str + "\n\n")
    readme.write("enviroment: " + env_name + "\n\n")
    readme.write("description: " + description + "\n\n")
    readme.write("time_elapsed: " + runtime_str + "\n\n")
    readme.write("num_timesteps: " + str(num_timesteps) + "\n\n")
    readme.write("seed: " + str(seed) + "\n\n")
    readme.close()

    # TODO add code snippets that correspond to the run
    # TODO somehow store the tensorboard logs here after the fact

    saver = tf.train.Saver()
    saver.save(tf.get_default_session(), save_dir + filename)

    os.rename("./tmp_logs/", save_dir + "tensorboard")

if __name__ == "__main__":

    name = "please enter a name for this run"
    description = input("please enter a description of the run")

    if name is None or description is None:
        raise Exception("you need to provide a name and description")

    start_time = time.time()

    env_name = "Acrobot-v1"
    # env_name = 'InvertedPendulumPyBulletEnv-v0'
    # env_name = "su_cartpole_et-v0"
    # env_name = "InvertedDoublePendulum-v2"

    num_timesteps = 2e6
    seed = 0

    logger.configure(dir="./tmp_logs", format_strs=["tensorboard"])
    with tf.device("/cpu:0"):
        pi = train(env_name, num_timesteps=num_timesteps, seed=seed)

    runtime = time.time() - start_time

