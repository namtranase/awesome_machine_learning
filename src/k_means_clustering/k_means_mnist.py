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

def process_data():
    """Process KNN program.
    """
    logging.debug('Centrel found by sklearn kmeans: %s', model.cluster_centers_)

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
