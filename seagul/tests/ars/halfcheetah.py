from seagul.rl.ars.ars_pipe import ars
from seagul.nn import MLP
import torch
import matplotlib.pyplot as plt
from seagul.mesh import variation_dim
import time
import copy
import gym

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

env_name = "Hopper-v2"
env = gym.make(env_name)
in_size = env.observation_space.shape[0]
out_size = env.action_space.shape[0]
policy = MLP(in_size,out_size,0,0)

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

def do_rollout(env, policy, render=False):
    torch.autograd.set_grad_enabled(False)

    act_list = []
    obs_list = []
    rew_list = []

    dtype = torch.float32
    obs = env.reset()
    done = False
    cur_step = 0

    while not done:
        obs = torch.as_tensor(obs, dtype=dtype).detach()
        obs_list.append(obs.clone())

        act = policy(obs)
        obs, rew, done, _ = env.step(act.numpy())

        if render:
            env.render()
            time.sleep(.02)

        act_list.append(torch.as_tensor(act.clone()))
        rew_list.append(rew)

        cur_step += 1

    ep_length = len(rew_list)
    ep_obs = torch.stack(obs_list)
    ep_act = torch.stack(act_list)
    ep_rew = torch.tensor(rew_list, dtype=dtype)
    ep_rew = ep_rew.reshape(-1, 1)

    torch.autograd.set_grad_enabled(True)
    return ep_obs, ep_act, ep_rew, ep_length, None

env = gym.make(env_name)
o,a,r,l,_ = do_rollout(env,Pc[0],render=True)