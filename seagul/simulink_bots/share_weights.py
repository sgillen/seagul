import numpy as np
import array


def read_weights(file_name, dim1, dim2):
    f = open(file_name, "rb")
    a = array.array("d")  # need to change d->f this for float32
    A = np.zeros((dim1, dim2))

    size = dim1 * dim2
    a.fromfile(f, size)
    for i in range(dim2):
        A[:, i] = a[i * dim1 : i * dim1 + dim1]

    return A


def write_weights(file_name, A):
    f = open(file_name, "wb")
    a = array.array("d")

    for entry in A.flatten():
        a.append(entry)

    a.tofile(f)
