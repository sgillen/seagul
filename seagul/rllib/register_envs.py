# Ray requires it's own registry, can't rely on the normal mechanisms that gym uses

from seagul.envs.mujoco.five_link import FiveLinkWalkerEnv
from pybullet_envs.gym_locomotion_envs import Walker2DBulletEnv
from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv
import gym
import pybullet_envs
import seagul.envs


from ray.tune.registry import register_env

def five_link_creator(env_config):
    return FiveLinkWalkerEnv()
    


def bullet_walker_creator(env_config):
#    return Walker2DBulletEnv()
    return gym.make("Walker2DBulletEnv-v0")

def bullet_humanoid_creator(env_config):
    #return HumanoidBulletEnv()
    return gym.make("HumanoidBulletEnv-v0")



def register_all_envs():
    register_env("five_link-v3", five_link_creator)
    register_env("Walker2DBulletEnv-v0", bullet_walker_creator)
    register_env("HumanoidBulletEnv-v0", bullet_humanoid_creator)

if __name__ == "__main__":
    register_all_envs()
