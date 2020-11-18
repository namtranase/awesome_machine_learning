from __future__ import print_function
import logging
import random
from time import time

import numpy as np
from scipy.spatial.distance import cdist

from sklearn.datasets import fetch_openml # fetch_mldata is dead
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

import settings
from src.config.config import read_config_file

def prepare_data(config):
    """Prepare data for training.
    """
    data_dir = config['data']['mnist']
    mnist = fetch_openml('mnist_784',
                         data_home=data_dir,
                         version=1,
                         cache=True)

    logging.debug("Shape of mnist data: ", mnist.data.shape)

    return mnist

def process_data(config):
    """Main program for Kmeans clustering for MNIST dataset.
    """
    logging.debug('Fetch mnist dataset...')
    mnist = prepare_data(config)

    # TODO: need to improve
    # Kmeans for Mnist data
    K = 10
    N = 10000
    X = mnist.data(np.random.choice(mnist.data.shape[0], N))
    kmeans = KMeans(n_clusters=K).fit(X)
    pred_label = kmeans.predict(X)

def main():
    """Main program for Kmeans clustering for MNIST dataset.
    """
    config = read_config_file(settings.config_file)
    if config['debug']:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug('Start K-means with config: %s', config)
    process_data(config)

if __name__ == "__main__":
    main()
