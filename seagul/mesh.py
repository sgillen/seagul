import numpy as np
import scipy.optimize as opt


def create_mesh(data, d, initial_mesh=None):
    """ Creates a mesh from the given data using balls of size d
    Args:
        data: np.array, the data you want to create a mesh for
        d: float, the radius for the ball used to determine membership in the mesh
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


def mesh_dim(data, d=1e-3):
    mesh_sizes = []
    d_vals = []
    while True:
        mesh, _ = create_mesh(data, d)
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


def conservative_mesh_dim(data, d=1e-3):
    mesh_sizes = []
    d_vals = []
    while True:
        mesh, _ = create_mesh(data, d)
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
