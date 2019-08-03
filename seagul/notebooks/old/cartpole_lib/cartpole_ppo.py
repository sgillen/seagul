from baselines.common.cmd_util import make_mujoco_env
from baselines.common import tf_util as U
from baselines import logger
from baselines.ppo1 import pposgd_simple

from cartpole.cartpole_sim import cartpole_policy


def train(env_id, num_timesteps, seed=0):
    U.make_session(num_cpu=1).__enter__()

    def policy_fn(name, ob_space, ac_space):
        return cartpole_policy.CartPolePolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=6,
                                              num_hid_layers=2)

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


if __name__ == '__main__':
    logger.configure(dir = "./tensorboard_test", format_strs=["tensorboard"] )
    pi = train('InvertedPendulum-v2', num_timesteps=5000, seed=0)
