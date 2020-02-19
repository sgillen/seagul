import matplotlib.pyplot as plt
import visualize

# path can be to multiple samples or inside a specific one:
checkpoint = "240" # only needed for rendering
# res_dir = ['./data/HalfCheetahBulletEnv-v0/debug_ddpg/DDPG/DDPG_HalfCheetahBulletEnv-v0_fa22b113_2020-02-18_13-45-28q4lgdx2f/']
res_dir = ["./data/HalfCheetahBulletEnv-v0/compare_mlp/SAC/", "./Results/HalfCheetah/SAC/MIXED_1_bug_in_rbf/"]

for dir in res_dir:
    visualize.plot_progress(dir, smoothing_factor=5, cutoff=60000)
plot = False
if plot:
    plt.show()
else:
    plt.savefig("Results/Halfcheetah_ddpg_debug_ddpg.png")

visualize.render(checkpoint, res_dir[0])