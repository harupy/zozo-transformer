import os
import shutil
from sklearn.preprocessing import normalize


def empty_dir(dir_path):
    for fname in os.listdir(dir_path):
        fpath = os.path.join(dir_path, fname)
        if os.path.isfile(fpath):
            os.unlink(fpath)
        elif os.path.isdir(fpath):
            shutil.rmtree(fpath)


def l2_norm(x):
    return normalize(x, norm='l2')
