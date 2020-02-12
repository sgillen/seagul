import matplotlib.pyplot as plt
import visualize

output_dirs = ["./HalfCheetah_test/SAC/", "./data/HalfCheetah/SAC_all/SAC_HalfCheetahBulletEnv-v0_48677418_2020-02-10_18-50-377zn3_x67/"]

for dir in output_dirs:
    visualize.plot_progress(dir, smoothing_factor=3)
plt.show()
# plt.savefig("name.png")

algos = {
    0: "A2C",
    1: "A3C",
    2: "APEX",
    3: "APPO",
    4: "DDPG",
    5: "IMPALA",
    6: "PG",
    7: "PPO",
    8: "SAC",
    9: "TD3",
    11: "ES"
}
envs = {
    0: "HumanoidBulletEnv-v0",
    1: "Walker2DBulletEnv-v0",
    2: "Pendulum-v0",
    3: "HalfCheetahBulletEnv-v0"
}

alg = algos[8]
current_env = envs[3]
checkpoint = "1365"
home_path = '/home/grabka/Documents/seagul/seagul/rllib/rllib_with_rbf/HalfCheetah_mixed/SAC/SAC_HalfCheetahBulletEnv-v0_48677418_2020-02-10_18-50-377zn3_x67/'
visualize.render(alg, current_env, checkpoint, home_path)