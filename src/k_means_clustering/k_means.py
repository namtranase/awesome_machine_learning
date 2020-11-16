from __future__ import print_function
import logging
from time import time

import numpy as np
import scipy.spatial.distance import cdist
import random

import settings
from src.config.config import read_config_file

def prepare_data():
    """Prepare data for training.
    """
    means = [[2, 2], [8, 3], [3, 6]]
    cov = [[1, 0], [0, 1]]
    N = 500
    X0 = np.random.multivariate_normal(mean[0], cov, N)
    X1 = np.random.multivariate_normal(mean[1], cov, N)
    X2 = np.random.multivariate_normal(mean[2], cov, N)

    X = np.concatenate((X0, X1, X2), axis=0)
    # Define 3 clusters
    K = 3
    original_labes = np.asarray([0]*N + [1]*N + [2]*N).T

def process_data():
    """Process KNN program.
    """
    np.random.seed(18)


def main():
    """Main program for KNN program.
    """
    config = read_config_file(settings.config_file)
    if config['debug']:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug('Start K-means with config: %s', config)
    process_data()

if __name__ == "__main__":
    main()