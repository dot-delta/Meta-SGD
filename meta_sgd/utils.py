import numpy as np


def data_points_sample(k):
    """ Generating data smaple point (x, y) pairs parmater k 
        implies number of (x, y) pairs to sample
    """
    arr = [0, 1]
    p = [.5, .5]

    x = np.random.rand(k, 50)
    y = np.random.choice(arr, size=k, p=p).reshape([-1, 1])
    return x, y