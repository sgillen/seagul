from gym.envs.registration import register

register(
    id='mj_su_cartpole-v0',
    entry_point='gym_ucsb.mujoco:MJSUCartPoleEnv',
)

register(
    id='mj_su_cartpole_sparse-v0',
    entry_point='gym_ucsb.mujoco:MJSUCartPoleSparseEnv',
)

register(
    id='mj_su_cartpole_et-v0',
    entry_point='gym_ucsb.mujoco:MJSUCartPoleEtEnv',
)

register(
    id='mj_su_cartpole_discrete-v0',
    entry_point='gym_ucsb.mujoco:MJSUCartPoleDiscreteEnv',
)

register(
    id='su_cartpole-v0',
    entry_point='gym_ucsb.classic_control:SUCartPoleEnv',
)

register(
    id='su_cartpole_discrete-v0',
    entry_point='gym_ucsb.classic_control:SUCartPoleDiscEnv'
)

register(
    id='su_pendulum-v0',
    entry_point='gym_ucsb.classic_control:SUPendulumEnv',
)

register(
    id='lorenz-v0',
    entry_point='gym_ucsb.simple_nonlinear:LorenzEnv',
)

register(
    id='five_link-v3',
    max_episode_steps=1000,
    entry_point='gym_ucsb.mujoco:FiveLinkWalkerEnv',
)

register(
    id='dyn_car-v0',
    entry_point='gym_ucsb.car:DynCarEnv',
)
