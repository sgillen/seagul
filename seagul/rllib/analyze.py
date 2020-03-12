import matplotlib.pyplot as plt
import numpy as np
import analysis_functions

# path can be to multiple samples or inside a specific one:
checkpoint = "3544" # only needed for rendering
# res_dir = ["./Results/HalfCheetah/PPO/FCN_1/PPO_HalfCheetahBulletEnv-v0_6dee9f88_2020-03-01_19-03-4581ly820s/"] # 180
# res_dir = ["./data/HalfCheetahBulletEnv-v0/PPO/"] # 10
res_dir = ["./Results/Walker2d-v2/ARS/"] # 31

entries = analysis_functions.outputs_to_df(res_dir, cutoff=-1)
analysis_functions.plot_outputs(entries)
env, alg = analysis_functions.get_params(res_dir)  
plt.title("Environment: " + env + ",  Algorithm: " + alg)

plot = False
if plot:
    plt.show()
    x=1
else:
    name = env + "_" + alg
    plt.savefig("./Results/" + name + ".png")

analysis_functions.render(checkpoint, res_dir[0])