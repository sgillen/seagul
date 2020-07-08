from seagul.rl.ars.ars_pipe import ars
from seagul.nn import MLP
import torch
import matplotlib.pyplot as plt
from seagul.mesh import variation_dim
import time
import copy


def shrinkdim(rews):
    return rews/variation_dim(rews)


def growdim(rews):
    return rews*variation_dim(rews)


start = time.time()
torch.set_default_dtype(torch.float64)

num_trials = 1
num_epochs = 100
n_workers = 8
n_delta = 32
n_top = 16

Ra = []
Pa = []
Rc = []
Pc = []
Rb = []
Pb = []
TRa = []
TRb = []
TRc = []

env_name = "HalfCheetah-v2"
policy = MLP(17,6,0,0)

for i in range(num_trials):
    policy, r_hist, lr_hist = ars(env_name, policy, num_epochs, n_workers=n_workers, n_delta=n_delta, n_top=n_top)
    print(i, time.time() - start)
    Pa.append(copy.deepcopy(policy))
    Ra.append(r_hist)
    TRa.append(lr_hist)
    plt.plot(lr_hist, 'k')
    # plt.show()

for i in range(num_trials):
    policy, r_hist, lr_hist = ars(env_name, policy, num_epochs, n_workers=n_workers, n_delta=n_delta, n_top=n_top, postprocess=shrinkdim)
    print(i, time.time() - start)
    Pb.append(copy.deepcopy(policy))
    Rb.append(r_hist)
    TRb.append(lr_hist)
    plt.plot(lr_hist, 'b')
    #plt.show()

for i in range(num_trials):
    policy, r_hist, lr_hist = ars(env_name, policy, num_epochs, n_workers=n_workers, n_delta=n_delta, n_top=n_top, postprocess=growdim)
    print(i, time.time() - start)
    Pc.append(copy.deepcopy(policy))
    Rc.append(r_hist)
    TRc.append(lr_hist)
    plt.plot(lr_hist, 'r')
    # plt.show()

plt.show()
print(time.time() - start)

