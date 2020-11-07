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

def process_data():
    """Process KNN program.
    """
    # N d-dimensional points
    X = np.random.randn(N, d)
    z = np.random.randn(d)
    logging.debug('Length of sample z: %s', len(z))

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
