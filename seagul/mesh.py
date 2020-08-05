import numpy as np
import copy
import scipy.optimize as opt
import torch
from collections.abc import MutableMapping


class MeshPoint:
    def __init__(self, identity, point):
        self.id = identity
        self.point = point
        self.freq = 1


class BylMesh(MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, d):
        self.d = d; self.scale = 1/d
        self.mesh = dict()

    def __getitem__(self, key):
        return self.mesh[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self.mesh[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.mesh[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.mesh)

    def __len__(self):
        return len(self.mesh)

    def __keytransform__(self, key):
        round_key = np.asarray(key)
        round_key = np.round(round_key * self.scale, decimals=0) * self.d
        round_key[round_key == -0.0] = 0.0
        round_key = tuple(round_key)
        return round_key



def create_mesh(data, d, initial_mesh=None):
    """ Creates a mesh from the given data using balls of size d
    Args:
        data: np.array, the data you want to create a mesh for
        d: float, the radius for the ball used to determine membership in the mesh
        initial_mesh: list, if you want to extend an existing mesh with new data, pass the old mesh as a list
    Returns:
        mesh: list, all the points from data that made it into the mesh
        weights: how many points from the original set are represented by the corresponding point in the mesh
    """
    if initial_mesh is None:
        initial_mesh = []

    mesh = initial_mesh
    weights = []
    in_mesh = np.zeros(data.shape[0], dtype=np.bool)

    for i, x in enumerate(data):
        if in_mesh[i]:
            continue
        else:
            in_criteria = np.linalg.norm(x - data, axis=1, ord=1) < d
            in_mesh = np.logical_or(in_mesh, in_criteria)
            mesh.append(x)
            weights.append(np.sum(in_criteria))

    return mesh, weights


def create_mesh_act(env, policy, d, seed_point, perturbs, reset_fn, snapshot_fn, interp_fn, ref_mean, ref_std):
    torch.autograd.set_grad_enabled(False)
    failure_point = np.ones_like(seed_point) * 10

    mesh = BylMesh(d)
    mesh[failure_point] = MeshPoint(0, failure_point)
    mesh[seed_point] = MeshPoint(1, seed_point)

    mesh_points = []
    mesh_points.append(failure_point)
    mesh_points.append(seed_point)

    transition_list = [[0] * len(perturbs)]  # Failure state always transitions to itself

    cnt = 0;
    cur_explored_cnt = 0
    for init_pos in mesh_points:
        if (init_pos == failure_point).all():
            continue

        cur_explored_cnt += 1
        transition_list.append([])

        tmp_points = []
        tmp_keys = []
        failed = False

        for pert in perturbs:
            if failed:
                break
            cnt += 1

            step = 0
            done = False;
            do_once = 1
            o = reset_fn(env, init_pos)

            while not done:
                a = policy(o) + do_once * pert
                step += 1
                do_once = 0

                last_o = copy.copy(o)

                o, r, done, _ = env.step(a.numpy())

                if snapshot_fn(o, last_o, step):

                    pt = interp_fn(o, last_o)
                    key = (pt - ref_mean) / ref_std

                    # weights = pca.singular_values_/pca.singular_values_.max()
                    # key = weights*pca.transform(key.reshape(1,-1)).reshape(-1)

                    tmp_points.append(copy.copy(pt))
                    tmp_keys.append(copy.copy(key))

                    if cnt % 1000 == 0:
                        print("explored: ", cur_explored_cnt, "| added: ", len(mesh), "| ratio: ",
                              cur_explored_cnt / len(mesh), "| failures: ", mesh[failure_point].freq, "| count: ", cnt)
                    done = True

                if step > 200:
                    failed = True
                    done = True

        if not failed:
            for p, k in zip(tmp_points, tmp_keys):
                if k in mesh:
                    mesh[k].freq += 1
                else:
                    mesh_points.append(p)
                    mesh[k] = MeshPoint(len(mesh_points)-1, p)

                transition_list[-1].append(mesh[k].id)


        else:
            for _ in range(len(perturbs)):
                mesh[failure_point].freq += 1
                transition_list[-1].append(mesh[failure_point].id)

    torch.autograd.set_grad_enabled(True)
    return mesh, mesh_points, np.array(transition_list)


def create_box_mesh(data, d, initial_mesh=None):
    """ Creates a mesh from the given data using boxes of size d
    Args:
        data: np.array, the data you want to create a mesh for
        d: float, the radius for the box used to determine membership in the mesh
    Returns:
        mesh: list, all the points from data that made it into the mesh
        weights: how many points from the original set are represented by the corresponding point in the mesh
    """
    if initial_mesh is None:
        initial_mesh = {}

    mesh = initial_mesh
    data = np.asarray(data)

    scale = 1/d

    keys = np.round(data*scale, decimals=0)*d
    keys[keys == -0.0] = 0.0

    for key in keys:
        key = tuple(key)
        if key in mesh:
            mesh[key] +=1
        else:
            mesh[key] = 1

    return mesh


def mesh_dim(data, scaling_factor=1.5, init_d=1e-2, upper_size_ratio=4/5, lower_size_ratio=0.0, d_limit=1e-9):
    """
    Args:
        data - any array like, represents the trajectory you want to compute the dimension of
        scaling factors - float indicating how much to scale d by for every new mesh
        init_d - float, initial box size
        upper_size_ratio - upper_size_ratio*data.shape[0] determines what size of mesh to stop at when finding the upper bound of the curve.
        lower_size_ratio - lower_size_ratio*data.shape[0] determines what size of mesh to stop at when finding the lower bound of the curve. Usually best to leave at 0.
        d_limit - smallest d value to allow when seeking the upper_size bound

    Returns:
        mdim: linear fit to the log(mesh) log(d) data, intentional underestimate of the meshing dimensions
        cdim: the conservative mesh dimension, that is the largest slope from the log log data, an intentional overestimate of the
        mesh_sizes: sizes of each mesh created during the computation
        d_vals: box sizes used to create each mesh during the computation
    """

    mesh_size_upper = np.round(upper_size_ratio * data.shape[0])
    mesh_size_lower = np.round(np.max((1.0, lower_size_ratio * data.shape[0])))
    d = init_d

    mesh = create_box_mesh(data, d)
    mesh_sizes = [len(mesh)]
    d_vals = [d]

    while mesh_sizes[0] < mesh_size_upper and d > d_limit:
        d /= scaling_factor
        mesh = create_box_mesh(data, d)
        mesh_sizes.insert(0, len(mesh))
        d_vals.insert(0, d)

    d = init_d
    while mesh_sizes[-1] > mesh_size_lower and d > d_limit:
        d = d * scaling_factor
        mesh = create_box_mesh(data, d)
        mesh_sizes.append(len(mesh))
        d_vals.append(d)

    for i, m in enumerate(mesh_sizes):
        if m < mesh_size_upper:
            lin_begin = i
            break

    xdata = np.log2(d_vals[lin_begin:])
    ydata = np.log2(mesh_sizes[lin_begin:])

    # Fit a curve to the log log line
    def f(x, m, b):
        return m * x + b

    popt, pcov = opt.curve_fit(f, xdata, ydata)

    # find the largest slope
    min_slope = 0
    for i in range(len(ydata) - 2):
        slope = (ydata[i+1] - ydata[i]) / (xdata[i + 1] - xdata[i])
        if slope < min_slope:
            min_slope = slope

    return -popt[0], -min_slope, mesh_sizes, d_vals


def power_var(X, l, ord):
    diffs = X[l:] - X[:-l]
    norms = np.zeros(diffs.shape[0])
    for i,d in enumerate(diffs):
        norms[i] = np.linalg.norm(d,ord=1)
        
    return 1 / (2 * len(X) - l) * np.sum(norms)


def variation_dim(X, order=1):
    # Implements the order p variation fractal dimension from https://arxiv.org/pdf/1101.1444.pdf (eq 18)
    # order 1 corresponds to the madogram, 2 to the variogram, 1/2 to the rodogram
    return 2 - 1/(order*np.log(2))*(np.log(power_var(X, 2, order)) - np.log(power_var(X, 1, order)))


# if __name__ == "__main__":
#     import numpy as np
#     import time
#     import matplotlib.pyplot as plt
#
#     data_size = 10000
#     meshes = []
#
#     dim = 17
#     coefs = np.random.random((dim,))
#     start = np.zeros((dim,))
#     stop = np.ones((dim)) * 10
#
#     x = np.linspace(start, stop, 100)
#     noise = np.random.random(x.shape)
#     data = coefs * (x + noise)
#
#     #X = np.((int(data_size), 10))
#     start = time.time()
#     m, c, mesh_sizes, d_vals = mesh_dim(data, .01)
#
#     xdata = np.log2(d_vals)
#     ydata = np.log2(mesh_sizes)
#
#     plt.plot(d_vals, mesh_sizes, 'x-')
#     plt.show()
#     plt.plot(xdata, ydata, 'x-')
#     plt.show()


if __name__ == "__main__":
    from seagul.rl.ars.ars_pipe import ars
    from seagul.nn import MLP
    import torch
    import matplotlib.pyplot as plt
    from seagul.mesh import create_mesh, variation_dim, mesh_dim, create_box_mesh
    import time
    import copy
    import gym
    import torch

    import xarray as xr
    import numpy as np
    import pandas as pd

    import os


    def identity(rews, obs, acts):
        return rews


    def vardiv(rews, obs, acts):
        return rews / variation_dim(obs)


    def varmul(rews, obs, acts):
        return rews * variation_dim(obs)


    def rough_measure(x):
        d = torch.stack([x[t, :] - x[t - 1, :] for t in range(1, x.shape[0])])
        return d.std()  # $/(torch.abs(d.mean()))


    def rmul(rews, obs, acts):
        return rews * rough_measure(obs)


    def rdiv(rews, obs, acts):
        return rews / rough_measure(obs)


    def pcastd(rews, obs, acts):
        pca = PCA()
        pca.fit((obs - obs.mean()) / obs.std())
        return rews * pca.explained_variance_ratio_.std()


    def mdim_mul(rews, obs, acts):
        m, _, _, _ = mesh_dim(obs)
        return m * rews


    def mdim_div(rews, obs, acts):
        m, _, _, _ = mesh_dim(obs)
        return rews / m


    def cdim_mul(rews, obs, acts):
        _, c, _, _ = mesh_dim(obs)
        return c * rews


    def cdim_div(rews, obs, acts):
        _, c, _, _ = mesh_dim(obs)
        return rews / c


    def madodiv(rews, obs, acts):
        return rews / variation_dim(obs, order=1)


    def variodiv(rews, obs, acts):
        return rews / variation_dim(obs, order=2)


    def radodiv(rews, obs, acts):
        return rews / variation_dim(obs, order=.5)


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
                time.sleep(.01)

            act_list.append(torch.as_tensor(act.clone()))
            rew_list.append(rew)

            cur_step += 1

        ep_length = len(rew_list)
        ep_obs = torch.stack(obs_list)
        ep_act = torch.stack(act_list)
        ep_rew = torch.tensor(rew_list, dtype=dtype)
        ep_rew = ep_rew.reshape(-1, 1)

        torch.autograd.set_grad_enabled(True)
        return ep_obs, ep_act, ep_rew, ep_length


    def do_long_rollout(env, policy, ep_length):
        torch.autograd.set_grad_enabled(False)

        act_list = []
        obs_list = []
        rew_list = []

        dtype = torch.float32
        obs = env.reset()
        done = False
        cur_step = 0

        while cur_step < ep_length:
            obs = torch.as_tensor(obs, dtype=dtype).detach()
            obs_list.append(obs.clone())

            act = policy(obs)
            obs, rew, done, _ = env.step(act.numpy())

            act_list.append(torch.as_tensor(act.clone()))
            rew_list.append(rew)

            cur_step += 1

        ep_length = len(rew_list)
        ep_obs = torch.stack(obs_list)
        ep_act = torch.stack(act_list)
        ep_rew = torch.tensor(rew_list, dtype=dtype)
        ep_rew = ep_rew.reshape(-1, 1)

        torch.autograd.set_grad_enabled(True)
        return ep_obs, ep_act, ep_rew, ep_length


    env_name = "HalfCheetah-v2"

    num_experiments = 2
    data = torch.load("/home/sgillen/work/lorenz/run_ars/data_mcshdim4/HalfCheetah-v2.xr")
    policy_dict = data.policy_dict

    exp_names = [fn.__name__ for fn in data.attrs['post_fns']]
    num_seeds = len(policy_dict[exp_names[0]])

    combined_curves = rewards = xr.DataArray(np.zeros((num_experiments, num_seeds, 1000)),
                                             dims=("post", "trial", "epoch"),
                                             coords={"post": exp_names})

    rews = data.rews  # /data.post_rews
    combined_curves[:, :, 750:] = rews

    data = torch.load("/home/sgillen/work/lorenz/run_ars/data17/HalfCheetah-v2.xr")
    policy_dict['identity'] = data.policy_dict['identity']

    rews = data.rews  # /data.post_rews
    combined_curves[:, :, :750] = rews.loc['madodiv']

    means = combined_curves.mean(dim="trial")
    stds = combined_curves.std(dim="trial")

    plt.plot(means.T)
    plt.legend(exp_names, loc='upper left')
    ci = stds

    for mean, c in zip(means, ci):
        plt.fill_between([t for t in range(len(mean))], (mean - c), (mean + c), alpha=.5)
    plt.title("Preprocessed Reward")
    plt.show()

    seed = 2
    ep_length = 10000
    policy = policy_dict['identity'][seed]
    env = gym.make(env_name)
    o, a, r, _ = do_long_rollout(env, policy, ep_length=ep_length)
    # o,a,r,l = do_rollout(env, policy, render=True)

    plt.plot(o)
    plt.show()

    target = o[200:]
    target = (target - policy.state_means) / policy.state_std
    # target = (target - target.mean(dim=0))/target.std(dim=0)

    print(sum(r))
    m, c, l, d = mesh_dim(target)
    print(m)
    print(c)

    # ==============
    policy = policy_dict['mdim_div'][seed]
    #% time
    o2, a2, r2, _ = do_long_rollout(env, policy, ep_length=ep_length)
    # o2,a2,r2,l2 = do_rollout(env, policy, render=True)
    plt.plot(o2)
    plt.figure()

    target2 = o2[200:]
    target2 = (target2 - policy.state_means) / policy.state_std
    # target2 = (target2 - target2.mean(dim=0))/target2.std(dim=0)

    # target = (target - policy.state_means)/policy.state_std

    print(sum(r2))
    m2, c2, l2, d2 = mesh_dim(target2)
    print(m2)
    print(c2)