from __future__ import print_function
import logging
import random
from time import time

import numpy as np
from scipy.spatial.distance import cdist

from sklearn.datasets import fetch_mldata
from sklearn.cluster import KMeans

import settings
from src.config.config import read_config_file

def prepare_data():
    """Prepare data for training.
    """

def process_data():
    """Process KNN program for MNIST dataset.
    """
    logging.debug('Centrel found by sklearn kmeans: %s', 10)

def main():
    """Main program for KNN program for MNIST dataset.
    """
    config = read_config_file(settings.config_file)
    if config['debug']:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug('Start K-means with config: %s', config)
    process_data()

if __name__ == "__main__":
    main()
