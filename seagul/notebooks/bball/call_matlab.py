# %%
from seagul.envs.matlab.bball_env import BBallEnv

# %%

env = BBallEnv()
env.reset()
obs, reward, done, obs_dict = env.step(4)

env.animate(obs_dict["tout"], obs_dict["xout"])