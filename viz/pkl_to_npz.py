import pickle
import numpy as np

import pickle as pkl
import gzip
import numpy
import scipy

bm_path = "body_models/smpl/neutral/model.pkl"
with open(bm_path, "rb") as f:
    try:
        data = pkl.load(f, encoding="latin1")
    except ModuleNotFoundError as e:
        if "chumpy" in str(e):
            message = ("Failed to load pickle file because "
                "chumpy is not installed.\n"
                "The original SMPL body model archives store some arrays as chumpy arrays, these are cast back to numpy arrays before use but it is not possible to unpickle the data without chumpy installed.")
            raise ModuleNotFoundError(message) from e
        else:
            raise e

def clean(x):
    if 'chumpy' in str(type(x)):
        return np.array(x)
    elif type(x) == scipy.sparse.csc.csc_matrix:
        return x.toarray()
    else:
        return x

hack_bm_path = "body_models/smpl/neutral/model.npz"
data = {k: clean(v) for k,v in data.items() if type(v)}
data = {k: v for k,v in data.items() if type(v) == np.ndarray}
np.savez(hack_bm_path, **data)