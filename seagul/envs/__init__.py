from gym.envs.registration import register


register(id="mj_su_cartpole-v0", entry_point="seagul.envs.mujoco:MJSUCartPoleEnv")
register(id="mj_su_cartpole_sparse-v0", entry_point="seagul.envs.mujoco:MJSUCartPoleSparseEnv")
register(id="mj_su_cartpole_et-v0", entry_point="seagul.envs.mujoco:MJSUCartPoleEtEnv")
register(id="mj_su_cartpole_discrete-v0", entry_point="seagul.envs.mujoco:MJSUCartPoleDiscreteEnv")
register(id="five_link-v3", max_episode_steps=1000, entry_point="seagul.envs.mujoco:FiveLinkWalkerEnv")
register(id="det_humanoid-v1", entry_point="seagul.envs.mujoco:DetHumanoidEnv", max_episode_steps=1000)
register(id="humanoid_long-v1", entry_point="gym.envs.mujoco:HumanoidEnv")

register(id="lorenz-v0", entry_point="seagul.envs.simple_nonlinear:LorenzEnv")
register(id="linear_z-v0", entry_point="seagul.envs.simple_nonlinear:LinearEnv")
register(id="gen_nonlin-v0", entry_point="seagul.envs.simple_nonlinear:GenEnv")

register(id="dyn_car-v0", entry_point="seagul.envs.car:DynCarEnv")
register(id="bullet_car-v0", entry_point="seagul.envs.bullet:RacecarGymEnv_v1")
register(id="bullet_car_ast-v0", entry_point="seagul.envs.bullet:RacecarGymEnvAst_v1")
# register(id="walker2d_five_link-v0", entry_point="seagul.envs.bullet:Walker2DFiveLink")

register(id="su_cartpole-v0", entry_point="seagul.envs.classic_control:SUCartPoleEnv")
register(id="sg_cartpole-v0", entry_point="seagul.envs.classic_control:SGCartPoleEnv")
register(id="su_cartpole_push-v0", entry_point="seagul.envs.classic_control:SUCartPolePushEnv")
register(id="su_cartpole_discrete-v0", entry_point="seagul.envs.classic_control:SUCartPoleDiscEnv")
register(id="su_pendulum-v0", entry_point="seagul.envs.classic_control:SUPendulumEnv")
register(id="su_acrobot-v0", entry_point="seagul.envs.classic_control:SGAcroEnv")


register(id="su_acrobot-v2", entry_point="seagul.envs.classic_control:SGAcroEnv2")
register(id="su_acroswitch-v0", entry_point="seagul.envs.classic_control:SGAcroSwitchEnv")
register(id="su_acroswitchsin-v0", entry_point="seagul.envs.classic_control:SGAcroSwitchSinEnv")
register(id="su_cartpole_gym-v0", entry_point="seagul.envs.classic_control:CartPoleEnv")
register(id="sym_pendulum-v0", entry_point="seagul.envs.classic_control:PendulumSymEnv", max_episode_steps=200)
register(id="dt_pendulum-v0", entry_point="seagul.envs.classic_control:PendulumDtEnv", max_episode_steps=200)
register(id="su_acro_drake-v0", entry_point="seagul.envs.drake:DrakeAcroEnv")

# Also go ahead and try to register environments for rllib as well

try:
    import pybullet_envs
except:
    import warnings
    warnings.warn("Warning, pybullet envs not installed")

try:
    import switched_rl.dm_gym
except:
    import warnings
    warnings.warn("Warning, pybullet envs not installed")

try:    
    # Ray requires it's own registry, can't rely on the normal mechanisms that gym uses
    
    #    from seagul.envs.mujoco.five_link import FiveLinkWalkerEnv
    import gym
    from ray.tune.registry import register_env

    #   def five_link_creator(env_config):
    #       return FiveLinkWalkerEnv()

    #TODO I'm sure we can find a way to register all envs currently in the registry automatically...
    def bullet_walker_creator(env_config):
        return gym.make("Walker2DBulletEnv-v0")

    def bullet_cheetah_creator(env_config):
        return gym.make("HalfCheetahBulletEnv-v0")

    def bullet_humanoid_creator(env_config):
        return gym.make("HumanoidBulletEnv-v0")

    def sym_pendulum_creator(env_config):
        return gym.make("sym_pendulum-v0")

    def dt_pendulum_creator(env_config):
        return gym.make("dt_pendulum-v0")

    def sg_pendulum_creator(env_config):
        return gym.make("sg_cartpole-v0")

    def humanoid_long_creator(env_config):
        return gym.make("humanoid_long-v1")

    def lorenz_creator(env_config):
        return gym.make("lorenz-v0", **env_config)

    def linear_creator(env_config):
        return gym.make("linear_z-v0", **env_config)

    def generic_creator(env_config):
        return gym.make("gen_nonlin-v0", **env_config)

    def drake_creator(env_config):
        return gym.make("su_acro_drake-v0", **env_config)

    def acro_creator(env_config):
        return gym.make("su_acrobot-v0", **env_config)

    def acroswitch_creator(env_config):
        return gym.make("su_acroswitch-v0", **env_config)
    
    def dm_creator(env_config):
        return gym.make("dm_acrobot-v0", **env_config)


    

    #  register_env("five_link-v3", five_link_creator)
    register_env("Walker2DBulletEnv-v0", bullet_walker_creator)
    register_env("HumanoidBulletEnv-v0", bullet_humanoid_creator)
    register_env("HalfCheetahBulletEnv-v0", bullet_cheetah_creator)
    register_env("sym_pendulum-v0", sym_pendulum_creator)
    register_env("dt_pendulum-v0", dt_pendulum_creator)
    register_env("sg_cartpole-v0", sg_pendulum_creator)
    register_env("humanoid_long-v1", humanoid_long_creator)
    register_env("lorenz-v0", lorenz_creator)
    register_env("linear_z-v0", linear_creator)
    register_env("gen_nonlin-v0", generic_creator)
    register_env("su_acro_drake-v0", drake_creator)
    register_env("su_acrobot-v0", acro_creator)
    register_env("su_acroswitch-v0", acroswitch_creator)
    register_env("dm_acrobot-v0", dm_creator)

except:
    import warnings
    warnings.warn("Warning, rllib environments not registered")
