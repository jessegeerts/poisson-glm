import numpy as np


def get_dir_from_xy(x, y):
     return np.arctan2(np.ediff1d(x, to_end=[0]), np.ediff1d(y, to_end=[0]))
