import numpy as np
import copy
import scipy.optimize as opt
import torch
from collections.abc import MutableMapping
import warnings

try:
    from stable_baselines3.common.callbacks import BaseCallback

    class RewardPostprocessCallback(BaseCallback):
        def __init__(self,
                     postprocessor,
                     verbose=0
                     ):
            super(RewardPostprocessCallback, self).__init__(verbose)
            self.postprocessor = postprocessor
            self.verbose = verbose

        def _on_rollout_end(self) -> None:
            buf = self.locals['rollout_buffer']

            d = buf.dones
            n_envs = d.shape[1]

            xz, yz = d.nonzero()
            done_lists = [[] for _ in range(n_envs)]

            for x, y in zip(xz,yz):
                done_lists[y].append(x)

            # for done_list in done_lists:
            #     done_list.append(d.shape[0])

            print("hello")
            for i, d_list in enumerate(done_lists):
                last_idx = 0
                for j,cur_idx in enumerate(d_list):
                    self.locals['rollout_buffer'].returns[last_idx:cur_idx, i] = self.postprocessor(buf.observations[last_idx:cur_idx, i,:], buf.actions[last_idx:cur_idx, i,:], buf.returns[last_idx:cur_idx, i])





        def _init_callback(self) -> None:
            pass

        def _on_step(self) -> bool:
            return True

except ImportError:
    warnings.warn("Warning, stable baselines 3 not installed, reward post processors won't be define")


class BoxMesh(MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, d):
        self.d = d;
        self.scale = 1 / d
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


def create_box_mesh(data, d, initial_mesh=None):
    """ Creates a mesh from the given data using boxes of size d
    Args:
        data: np.array, the data you want to create a mesh for
        d: float, the length of the box used to determine membership in the mesh
        initial_mesh: dict, output from a previous call to create_box_mesh, used to add to a mesh rather than build a new one
    Returns:
        mesh: dict, keys are the mesh point coordinates, values are how many points in the original data set are represented by the mesh point
    """
    if initial_mesh is None:
        initial_mesh = {}

    mesh = initial_mesh
    data = np.asarray(data)

    scale = 1 / d

    keys = np.round(data * scale, decimals=0) * d
    keys[keys == -0.0] = 0.0

    for key in keys:
        key = tuple(key)
        if key in mesh:
            mesh[key] += 1
        else:
            mesh[key] = 1

    return mesh


# Dimensionality calculations ==================================================
def mesh_dim(data, scaling_factor=1.2, init_d=1, upper_size_ratio=1.0, lower_size_ratio=0.0, d_limit=1e-9):
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


def variation_dim(X, order=1):
    # Implements the order p variation fractal dimension from https://arxiv.org/pdf/1101.1444.pdf (eq 18)
    # order 1 corresponds to the madogram, 2 to the variogram, 1/2 to the rodogram
    return 2 - 1 / (order * np.log(2)) * (np.log(power_var(X, 2, order)) - np.log(power_var(X, 1, order)))


def power_var(X, l, ord):
    # The power variation, see variation_dim
    diffs = X[l:] - X[:-l]
    norms = np.zeros(diffs.shape[0])
    for i, d in enumerate(diffs):
        norms[i] = np.linalg.norm(d, ord=1)

    return 1 / (2 * len(X) - l) * np.sum(norms)



def mesh_find_target_d(data, max_d_guess=10, target_size_ratio=1/2, d_lower_limit=1e-10, interval_target = 1e-6):
    """
    Find the first point d such that the mesh size for data is < target_size_ratio * d
    """
    max_mesh_size = data.shape[0]
    target_mesh_size = np.round(target_size_ratio * data.shape[0])

    max_mesh = create_box_mesh(data, d_lower_limit)
    max_mesh_size = len(max_mesh)
    if (max_mesh_size < target_mesh_size):
        print(f"warning: mesh size at d lower limit is {max_mesh_size} but target size is {target_mesh_size}, returning the lower limit")
        return d_lower_limit
    
    d = max_d_guess
    min_mesh = create_box_mesh(data, d)
    min_mesh_size = len(min_mesh)
    
    while min_mesh_size != 1:
        d = d*2
        min_mesh = create_box_mesh(data, d)
        min_mesh_size = len(min_mesh)

    d_upper_limit = d

    while (d_upper_limit - d_lower_limit) > interval_target:
        d = (d_upper_limit + d_lower_limit) / 2
        mesh = create_box_mesh(data,d)
        if (len(mesh) < target_mesh_size):
            d_upper_limit = d
        else:
            d_lower_limit = d

    return (d_upper_limit + d_lower_limit) / 2
            
        
        

# Post processors ==================================================
def identity(rews, obs, acts):
    return rews

def madodiv(rews, obs, acts):
    return rews/variation_dim(obs, order=1)

def variodiv(rews, obs, acts):
    return rews/variation_dim(obs, order=2)

def radodiv(rews, obs, acts):
    return rews/variation_dim(obs, order=.5)


def mdim_div(obs, acts, rews):
    if obs.shape[0] == 1000:
        gait_start = 200
        m, _, _, _ = mesh_dim(obs[gait_start:])
        m = np.clip(m, 1, obs.shape[1] / 2)
    else:
        m = obs.shape[1] / 2

    return rews / m

def target_d_div(obs,acts,rews):
    if obs.shape[0] == 1000:
        gait_start = 200
        target_d = mesh_find_target_d(obs[gait_start:])
    else:
        target_d = 10

    return rews / target_d


def target_d_divn(obs,acts,rews):
    if obs.shape[0] == 1000:
        gait_start = 200
        target_d = mesh_find_target_d(obs[gait_start:])
    else:
        target_d = 10

    return rews * (-np.log(target_d) + 20.0) / 20.0, 



def mdim_div2(obs_list, act_list, rew_list):
    combined_obs = torch.empty(0)
    combined_rew = torch.empty(0)
    m = None

    for obs, rew in zip(obs_list, rew_list):
        if obs.shape[0] == 1000:
            gait_start = 200
            combined_obs = torch.cat((combined_obs, obs[gait_start:]))
            combined_rew = torch.cat((combined_rew, rew))
        else:
            m = obs.shape[1] / 2

    if m is None:
        m, _, _, _ = mesh_dim(combined_obs)
        m = np.clip(m, 1, obs.shape[1] / 2)

    return (combined_rew / m).sum()

def mdim_div_stable(obs, act, rew):
    m = None

    if obs.shape[0] == 1000:
        gait_start = 200
        target_obs = obs[gait_start:]
    else:
        m = obs.shape[1] / 2

    if m is None:
        m, _, _, _ = mesh_dim(target_obs)
        m = np.clip(m, 1, obs.shape[1] / 2)

    return (rew / m)



if __name__ == "__main__":
    import numpy as np
    import time
    import matplotlib.pyplot as plt

    data_size = 10000
    meshes = []

    dim = 17
    coefs = np.random.random((dim,))
    start = np.zeros((dim,))
    stop = np.ones((dim)) * 10

    x = np.linspace(start, stop, 100)
    noise = np.random.random(x.shape)
    data = coefs * (x + noise)

    #X = np.((int(data_size), 10))
    start = time.time()
    m, c, mesh_sizes, d_vals = mesh_dim(data, .01)

    xdata = np.log2(d_vals)
    ydata = np.log2(mesh_sizes)

    plt.plot(d_vals, mesh_sizes, 'x-')
    plt.show()
    plt.plot(xdata, ydata, 'x-')
    plt.show()
