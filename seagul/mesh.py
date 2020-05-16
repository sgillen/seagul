import numpy as np

def create_mesh(data, d, initial_mesh=[]):
    """ Creates a mesh from the given data using balls of size d
    Args:
        data: np.array, the data you want to create a mesh for
        d: float, the radius for the ball used to determine membership in the mesh
    Returns:
        mesh: list, all the points from data that made it into the mesh
    """
    mesh = initial_mesh
    in_mesh = np.zeros(data.shape[0], dtype=np.bool)

    for i, x in enumerate(data):
        if in_mesh[i]:
            continue
        else:
            mesh.append(x)
            in_criteria = np.linalg.norm(mesh[-1] - data, axis=1, ord=1) < d
            in_mesh = np.logical_or(in_mesh, in_criteria)

    return mesh
            
         
