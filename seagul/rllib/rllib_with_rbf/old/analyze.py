# import matplotlib.pyplot as plt
import visualize

# path can be to multiple samples or inside a specific one:
checkpoint = "500" # only needed for rendering
# res_dir = ['./data/HalfCheetahBulletEnv-v0/debug_ddpg/DDPG/DDPG_HalfCheetahBulletEnv-v0_fa22b113_2020-02-18_13-45-28q4lgdx2f/']
res_dir = ["./seagul/seagul/rllib/rllib_with_rbf/SAC/SAC_HumanoidBulletEnv-v0_62a7a43c_2020-02-18_16-25-528ucxifxi/"]

# for dir in res_dir:
    # visualize.plot_progress(dir, smoothing_factor=2, cutoff=-1)
# plot = True
# if plot:
#     plt.show()
# else:
#     plt.savefig("Results/Halfcheetah_td3.png")

visualize.render(checkpoint, res_dir[0])
