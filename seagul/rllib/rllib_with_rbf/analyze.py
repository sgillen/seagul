import matplotlib.pyplot as plt
import numpy as np
import analysis_functions

# path can be to multiple samples or inside a specific one:
checkpoint = "20" # only needed for rendering
# res_dir = ['./data/HalfCheetahBulletEnv-0/debug_ddpg/DDPG/DDPG_HalfCheetahBulletEnv-v0_fa22b113_2020-02-18_13-45-28q4lgdx2f/']
res_dir = ["./data/HalfCheetahBulletEnv-v0/PPO/"] #ES_HumanoidBulletEnv-v0_dd3a4f36_2020-02-19_13-36-18xbj0wfpg/"]

entries = analysis_functions.outputs_to_df(res_dir, cutoff=-1)
analysis_functions.plot_outputs(entries)
env, alg = analysis_functions.get_params(res_dir)
plt.title("Environment: " + env + ",  Algorithm: " + alg)

plot = True
if plot:
    plt.show()
else:
    name = env + "_" + alg
    plt.savefig("./Results/" + name + ".png")

analysis_functions.render(checkpoint, res_dir[0])