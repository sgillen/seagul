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


def create_mesh_dict(data, d, initial_mesh=None):
    """ Creates a mesh from the given data using balls of size d
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



def mesh_dim(data, init_d=1e-6):
    """
    Args:
        data - any np array or torch thing
        init_d - initial mesh size

    Returns:
        mesh_dim, mesh_sizes, d_vals

    """
    scale_factor = 2

    mesh = create_mesh_dict(data, init_d)
    mesh_sizes = [len(mesh)]
    d_vals = [init_d]
    if len(mesh) != data.shape[0]:
        print("Warning initial d for mesh too large! auto adjusting")

        d = init_d/scale_factor
        while True:
            mesh = create_mesh_dict(data, d)
            mesh_sizes.insert(0, len(mesh))
            d_vals.insert(0, d)

            d = d/scale_factor

            if mesh_sizes[0] == data.shape[0]:
                break

    d = init_d*scale_factor
    while True:
        mesh = create_mesh_dict(data, d)
        mesh_sizes.append(len(mesh))
        d_vals.append(d)

        if mesh_sizes[-1] == 1:
            break

        d = d * scale_factor

    for i, m in enumerate(mesh_sizes):
        if m < data.shape[0]:
            lin_begin = i
            break

    xdata = np.log2(d_vals[lin_begin:])
    ydata = np.log2(mesh_sizes[lin_begin:])

    # Fit a curve to the log log line
    def f(x, m, b):
        return m * x + b

    popt, pcov = opt.curve_fit(f, xdata, -ydata)

    # find the largest slope
    min_slope = 0
    for i in range(len(ydata) - 2):
        slope = (ydata[i+1] - ydata[i]) / (xdata[i + 1] - xdata[i])
        if slope < min_slope:
            min_slope = slope

    return popt[0], -min_slope, mesh_sizes, d_vals


def conservative_mesh_dim(data, init_d=1e-3):
    """
    Args:
        data - any np array or torch thing
        init_d - initial mesh size

    Returns:
        mesh_dim, mesh_sizes, d_vals

    """
    mesh_sizes = []
    d_vals = []
    d = init_d
    while True:
        mesh = create_mesh_dict(data, d)
        mesh_sizes.append(len(mesh))
        d_vals.append(d)

        if mesh_sizes[-1] == 1:
            break

        d = d * 2



    min_slope = 0
    for i in range(len(mesh_sizes)-1):
        slope = - mesh_sizes[i] - mesh_sizes[i+1]/(d_vals[i+1] - d_vals[i])
        if slope < min_slope:
            min_slope = slope

    return -min_slope, mesh_sizes, d_vals


def power_var(X, l, ord):
    # Implements the power variation, used for the variation fractal dimension
    return 1 / (2 * len(X) - l) * np.sum(np.linalg.norm(X[l:] - X[:-l],ord=ord))


def variation_dim(X, order=1):
    # Implements the order p variation fractal dimension from https://arxiv.org/pdf/1101.1444.pdf (eq 18)
    # order 1 corresponds to the madogram, 2 to the variogram, 1/2 to the rodogram
    return 2 - 1/(order*np.log(2))*(np.log(power_var(X, 2, order)) - np.log(power_var(X, 1, order)))


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