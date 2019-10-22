from gym.envs.registration import register

register(id="mj_su_cartpole-v0", entry_point="seagul.envs.mujoco:MJSUCartPoleEnv")
#register(id="five_link")

register(id="mj_su_cartpole_sparse-v0", entry_point="seagul.envs.mujoco:MJSUCartPoleSparseEnv")

register(id="mj_su_cartpole_et-v0", entry_point="seagul.envs.mujoco:MJSUCartPoleEtEnv")

register(id="mj_su_cartpole_discrete-v0", entry_point="seagul.envs.mujoco:MJSUCartPoleDiscreteEnv")

register(id="su_cartpole-v0", entry_point="seagul.envs.classic_control:SUCartPoleEnv")
register(id="su_cartpole-v1", entry_point="seagul.envs.classic_control:SUCartPoleEnv1")
register(id="su_cartpole-v2", entry_point="seagul.envs.classic_control:SUCartPoleEnv2")
register(id="su_cartpole-v3", entry_point="seagul.envs.classic_control:SUCartPoleEnv3")


register(id="su_cartpole_push-v0", entry_point="seagul.envs.classic_control:SUCartPolePushEnv")

register(id="su_cartpole_discrete-v0", entry_point="seagul.envs.classic_control:SUCartPoleDiscEnv")

register(id="su_pendulum-v0", entry_point="seagul.envs.classic_control:SUPendulumEnv")

register(id="lorenz-v0", entry_point="seagul.envs.simple_nonlinear:LorenzEnv")

register(id="five_link-v3", max_episode_steps=1000, entry_point="seagul.envs.mujoco:FiveLinkWalkerEnv")

register(id="dyn_car-v0", entry_point="seagul.envs.car:DynCarEnv")

register(id="bullet_car-v0", entry_point="seagul.envs.bullet:RacecarGymEnv_v1")

register(id="bullet_car_ast-v0", entry_point="seagul.envs.bullet:RacecarGymEnvAst_v1")

register(id="su_acrobot-v0" , entry_point="seagul.envs.classic_control:AcrobotEnv")
register(id="su_acrobot-v1" , entry_point="seagul.envs.classic_control:AcrobotEnv2")


register(id="su_cartpole_gym-v0" , entry_point="seagul.envs.classic_control:CartPoleEnv")

#register(id="walker2d_five_link-v0", entry_point="seagul.envs.bullet:Walker2DFiveLink")

register(id="su_acro_drake-v0", entry_point="seagul.envs.drake:DrakeAcroEnv")
