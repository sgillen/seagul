import matplotlib.pyplot as plt
import numpy as np
import analysis_functions

# path can be to multiple samples (1) or inside a specific one (2):
# (1) res_dir = ["./data/HalfCheetahBulletEnv-v0/PPO/"]
# (2) res_dir = ["./data/HalfCheetahBulletEnv-v0/PPO/FCN_1/PPO_HalfCheetahBulletEnv-v0_6dee9f88_2020-03-01_19-03-4581ly820s/"]

checkpoint = "3544" # only needed for rendering
res_dir = ["./data/HalfCheetahBulletEnv-v0/PPO/"]

entries = analysis_functions.outputs_to_df(res_dir, cutoff=-1)
analysis_functions.plot_outputs(entries)
env, alg = analysis_functions.get_params(res_dir)  
plt.title("Environment: " + env + ",  Algorithm: " + alg)

plot = True # True-> plot without saving png, False-> save png to directory without plotting
if plot:
    plt.show()
else:
    name = env + "_" + alg
    plt.savefig("./Results/" + name + ".png")

# Rendering a specific checkpoint: (comment out to compare multiple trials)
analysis_functions.render(checkpoint, res_dir[0])
