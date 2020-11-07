from __future__ import print_function
import logging
from time import time

import numpy as np
import pandas as pd

import settings
from src.config.config import read_config_file

# d: dimension of sample, N: number of sample
d = 1000
N = 10000

def dist_pp(z, x):
    """Calculate norm l2 btw two vectors.
    """
    if not z.all() or not x.all():
        return None

    d = z - x.reshape(z.shape)

    return np.sum(d*d)

def dist_ps_naive(z, X):
    """Calculate norm l2 btw z and matrix Z
    based on dist_pp.
    """
    if not z.all() or not X.all():
        return None

    N = X.shape[0]
    res = np.zeros((1, N))
    logging.debug('Type of results: %s', res.shape)
    for i in range(N):
        res[0][i] = dist_pp(z, X[i])

    return res

def dist_ps_fast(z, X):
    """Calculate distance btw two vectors by
    the smart way.
    """
    if not z.all() or not X.all():
        return None
    # Square of l2 norm of each row of X
    X2 = np.sum(X*X, 1)
    z2 = np.sum(z*z)
    # z2 can be ignore
    return X2 + z2 - 2*X.dot(z)

def process_data():
    """Process KNN program.
    """
    # N d-dimensional points
    X = np.random.randn(N, d)
    z = np.random.randn(d)
    logging.debug('Length of sample z: %s', len(z))

    t1 = time()
    D1 = dist_ps_naive(z, X)
    logging.debug("Regular way knn, running time: %s s", time() - t1)

    t2 = time()
    D2 = dist_ps_fast(z, X)
    logging.debug("Fast way knn, running time: %s s", time() - t2)
    logging.debug("Results diffirence: %s", np.linalg.norm(D1 - D2))

def main():
    """Main program for KNN program.
    """
    config = read_config_file(settings.config_file)
    if config['debug']:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug('Start KNN with config: %s', config)
    process_data()

if __name__ == "__main__":
    main()
