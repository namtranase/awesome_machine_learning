from __future__ import print_function
import logging
from time import time

import numpy as np
from scipy.spatial.distance import cdist
import random
from sklearn.cluster import KMeans

import settings
from src.config.config import read_config_file

def prepare_data():
    """Prepare data for training.
    """

    return None

def process_data():
    """Process KNN program.
    """
    np.random.seed(18)
    # Define 3 clusters
    K = 3

    # Prepare data
    X, original_labels = prepare_data()
    logging.debug('Numbers of data: %s', len(X))
    logging.debug('Labels of data: %s', set(original_labels))

    # Simple kmeans
    (centroids, labels, it) = kmeans(X, K)
    logging.debug('Centrel found by simple kmeans: %s', centroids[-1])

    # scikit-learn kmeans
    model = KMeans(n_clusters=3, random_state=0).fit(X)
    logging.debug('Centrel found by sklearn kmeans: %s', model.cluster_centers_)

def main():
    """Main program for Naive bayes program.
    """
    config = read_config_file(settings.config_file)
    if config['debug']:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug('Start K-means with config: %s', config)
    process_data()

if __name__ == "__main__":
    main()
