import numpy as np
import scipy.optimize as opt
import collections


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


from collections.abc import MutableMapping


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


# class BylMesh:
#     def __init__(self,d):
#         self.d = d; self.scale = 1/d
#         self.mesh = collections.OrderedDict()
#
#     def __getitem__(self, item):
#         round_key = self.round_key(item)
#         return self.mesh[round_key]
#
#     def __setitem__(self, key, value):
#         round_key = self.round_key(key)
#         self.mesh[round_key] = value
#
#     def round_key(self, key):
#         round_key = np.asarray(key)
#         round_key = np.round(round_key * self.scale, decimals=0) * self.d
#         round_key[round_key == -0.0] = 0.0
#         round_key = tuple(round_key)
#         return round_key


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



def mesh_dim(data, init_d=1e-3):
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

    for i, m in enumerate(mesh_sizes):
        if m < data.shape[0]:
            lin_begin = i + 2
            break

    xdata = np.array(d_vals[lin_begin:])
    ydata = np.array(mesh_sizes[lin_begin:])

    def f(x, m, b):
        return m * x + b

    popt, pcov = opt.curve_fit(f, np.log10(xdata), -np.log10(ydata))
    return popt[0], mesh_sizes, d_vals


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

    max_slope = 0
    for i in range(len(mesh_sizes)-1):
        slope = - mesh_sizes[i] - mesh_sizes[i+1]/(d_vals[i+1] - d_vals[i])
        if slope > max_slope:
            max_slope = slope

    return max_slope, mesh_sizes, d_vals


def power_var(X, l, ord):
    # Implements the power variation, used for the variation fractal dimension
    return 1 / (2 * len(X) - l) * np.sum(np.linalg.norm(X[l:] - X[:-l],ord=ord))


def variation_dim(X, order=1):
    # Implements the order p variation fractal dimension from https://arxiv.org/pdf/1101.1444.pdf (eq 18)
    # order 1 corresponds to the madogram, 2 to the variogram, 1/2 to the rodogram
    return 2 - 1/(order*np.log(2))*(np.log(power_var(X, 2, order)) - np.log(power_var(X, 1, order)))



if __name__ == "__main__":
    from seagul.mesh import BylMesh
    import numpy as np

    m = BylMesh(.1)
    a = np.random.random(4)
    m[a] = 1

    print(a in m)
#
#
# if __name__ == "__main__":
#     import numpy as np
#     import time
#     import matplotlib.pyplot as plt
#
#     data_sizes = np.logspace(0,4.5)
#     meshes = []
#     dict_meshes = []
#     times = np.zeros_like(data_sizes, dtype=np.float)
#     dict_times = np.zeros_like(data_sizes, dtype=np.float)
#
#     for i, data_size in enumerate(data_sizes):
#         print(data_size)
#         X = np.random.random((int(data_size),5))
#         #X = np.ones((2000,5))
#         start = time.time()
#         #m,w = create_mesh(X, .01)
#         times[i] = time.time() - start
#         #meshes.append(m)
#
#         start = time.time()
#         m = create_mesh_dict(X,2)
#         dict_times[i] = time.time() - start
#         dict_meshes.append(m)
#
#     plt.title("Meshing time vs number of data points")
#    # plt.plot(data_sizes,times, '-o')
#     plt.plot(data_sizes, dict_times, '-o')
#     plt.plot(data_sizes, dict_times, '-o')
#
#     #plt.legend(['b'])
#     plt.ylabel("Time (s)")
#     plt.xlabel("Data set size")
#     plt.show()
#     #plt.legend('a','b')