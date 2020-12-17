from __future__ import print_function
from sklearn.naive_bayes import MutinomialNB
import logging
from time import time

import numpy as np

import settings
from src.config.config import read_config_file

def prepare_data(config):
    """Prepare data for training.
    """
    d1 = [2, 1, 1, 0, 0, 0, 0, 0, 0]
    d2 = [1, 1, 0, 1, 1, 0, 0, 0, 0]
    d3 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
    d4 = [0, 1, 0, 0, 0, 0, 1, 1, 1]

    train_data = np.array([d1, d2, d3, d4])
    labels = np.array('N', 'N', 'N' 'B')
    logging.debug("Number of samples: %s", len(train_data))

    return train_data, labels

def process_data(config):
    """Process KNN program.
    """
    # Prepare train data and their lables
    train_data, labels = prepare_data()

    # # Prepare data
    # X, original_labels = prepare_data()
    # logging.debug('Numbers of data: %s', len(X))
    # logging.debug('Labels of data: %s', set(original_labels))

    # # Simple kmeans
    # (centroids, labels, it) = kmeans(X, K)
    # logging.debug('Centrel found by simple kmeans: %s', centroids[-1])

    # # scikit-learn kmeans
    # model = KMeans(n_clusters=3, random_state=0).fit(X)
    # logging.debug('Centrel found by sklearn kmeans: %s', model.cluster_centers_)

def main():
    """Main program for Naive bayes program.
    """
    config = read_config_file(settings.config_file)
    if config['debug']:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug('Start K-means with config: %s', config)
    process_data(config)

if __name__ == "__main__":
    main()
