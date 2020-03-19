import numpy as np


def numpy_searchsorted(a: np.ndarray, v: np.ndarray, side='left'):
    """Numpy version of searchsorted that works batch-wise on pytorch tensors
    """
    nrows_a = a.shape[0]
    (nrows_v, ncols_v) = v.shape
    nrows_out = max(nrows_a, nrows_v)
    out = np.empty((nrows_out, ncols_v), dtype=np.long)
    def sel(data, row):
        return data[0] if data.shape[0] == 1 else data[row]
    for row in range(nrows_out):
        out[row] = np.searchsorted(sel(a, row), sel(v, row), side=side)
    return out
