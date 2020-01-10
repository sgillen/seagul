from gym.envs.registration import register

register(id="mj_su_cartpole-v0", entry_point="seagul.envs.mujoco:MJSUCartPoleEnv")
register(id="mj_su_cartpole_sparse-v0", entry_point="seagul.envs.mujoco:MJSUCartPoleSparseEnv")
register(id="mj_su_cartpole_et-v0", entry_point="seagul.envs.mujoco:MJSUCartPoleEtEnv")
register(id="mj_su_cartpole_discrete-v0", entry_point="seagul.envs.mujoco:MJSUCartPoleDiscreteEnv")
register(id="five_link-v3", max_episode_steps=1000, entry_point="seagul.envs.mujoco:FiveLinkWalkerEnv")
register(id="det_humanoid-v1", entry_point="seagul.envs.mujoco:DetHumanoidEnv", max_episode_steps=1000)
register(id="humanoid_long-v1", entry_point="gym.envs.mujoco:HumanoidEnv")

register(id="lorenz-v0", entry_point="seagul.envs.simple_nonlinear:LorenzEnv")

register(id="dyn_car-v0", entry_point="seagul.envs.car:DynCarEnv")
register(id="bullet_car-v0", entry_point="seagul.envs.bullet:RacecarGymEnv_v1")
register(id="bullet_car_ast-v0", entry_point="seagul.envs.bullet:RacecarGymEnvAst_v1")
# register(id="walker2d_five_link-v0", entry_point="seagul.envs.bullet:Walker2DFiveLink")

register(id="su_cartpole-v0", entry_point="seagul.envs.classic_control:SUCartPoleEnv")
register(id="sg_cartpole-v0", entry_point="seagul.envs.classic_control:SGCartPoleEnv")
register(id="su_cartpole_push-v0", entry_point="seagul.envs.classic_control:SUCartPolePushEnv")
register(id="su_cartpole_discrete-v0", entry_point="seagul.envs.classic_control:SUCartPoleDiscEnv")
register(id="su_pendulum-v0", entry_point="seagul.envs.classic_control:SUPendulumEnv")
register(id="su_acrobot-v0", entry_point="seagul.envs.classic_control:AcrobotEnv")
register(id="su_acrobot-v1", entry_point="seagul.envs.classic_control:AcrobotEnv2")
register(id="su_cartpole_gym-v0", entry_point="seagul.envs.classic_control:CartPoleEnv")
register(id="sym_pendulum-v0", entry_point="seagul.envs.classic_control:PendulumSymEnv", max_episode_steps=200)
register(id="dt_pendulum-v0", entry_point="seagul.envs.classic_control:PendulumDtEnv", max_episode_steps=200)
register(id="su_acro_drake-v0", entry_point="seagul.envs.drake:DrakeAcroEnv")

# Also go ahead and try to register environments for rllib as well
try:
    # Ray requires it's own registry, can't rely on the normal mechanisms that gym uses

    #    from seagul.envs.mujoco.five_link import FiveLinkWalkerEnv
    import gym
    import pybullet_envs

    from ray.tune.registry import register_env

    #   def five_link_creator(env_config):
    #       return FiveLinkWalkerEnv()

    def bullet_walker_creator(env_config):
        #    return Walker2DBulletEnv()
        return gym.make("Walker2DBulletEnv-v0")

    def bullet_humanoid_creator(env_config):
        # return HumanoidBulletEnv()
        return gym.make("HumanoidBulletEnv-v0")

    def sym_pendulum_creator(env_config):
        # return HumanoidBulletEnv()
        return gym.make("sym_pendulum-v0")

    def dt_pendulum_creator(env_config):
        # return HumanoidBulletEnv()
        return gym.make("dt_pendulum-v0")

    def sg_pendulum_creator(env_config):
        # return HumanoidBulletEnv()
        return gym.make("sg_cartpole-v0")

    def humanoid_long_creator(env_config):
        return gym.make("humanoid_long-v1")

    #  register_env("five_link-v3", five_link_creator)
    register_env("Walker2DBulletEnv-v0", bullet_walker_creator)
    register_env("HumanoidBulletEnv-v0", bullet_humanoid_creator)
    register_env("sym_pendulum-v0", sym_pendulum_creator)
    register_env("dt_pendulum-v0", dt_pendulum_creator)
    register_env("sg_cartpole-v0", sg_pendulum_creator)
    register_env("humanoid_long-v1", humanoid_long_creator)

except:
    import warnings

    warnings.warn("Warning, registering environments for rllib failed!")
