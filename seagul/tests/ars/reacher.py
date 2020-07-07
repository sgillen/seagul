from seagul.rl.ars.ars_pool import ars
from seagul.nn import MLP
import torch
import matplotlib.pyplot as plt
import gym.envs.mujoco.reacher


torch.set_default_dtype(torch.float64)

net = MLP(17,6,64,2)

net, r = ars("HalfCheetah-v2", net, 100)
plt.plot(r)
plt.show()