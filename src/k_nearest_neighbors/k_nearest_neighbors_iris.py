from __future__ import print_function
import logging
from time import time

import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import settings
from src.config.config import read_config_file

def split_dataset(config, test_size=130):
    """Get iris dataset and split to train, test set.
    """
    # Load dataset
    np.random.seed(7)
    iris = datasets.load_iris()
    iris_X = iris.data
    iris_y = iris.target
    logging.debug('Length of dataset: %s', len(iris_X))
    logging.debug('Lables of dataset: %s', np.unique(iris_y))
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        iris_X, iris_y, test_size=test_size)
    logging.debug("Train size: %s, Test size: %s",
                  X_train.shape[0], X_test.shape[0])

    return X_train, y_train, X_test, y_test

def process_data(config):
    """Process KNN program.
    """
    # Get dataset and split to train, test
    X_train, y_train, X_test, y_test = split_dataset(config)
    logging.debug('Length of sample z: %s', len('dfdfd'))

def main():
    """Main program for KNN for iris dataset program.
    """
    config = read_config_file(settings.config_file)
    if config['debug']:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug('Start KNN for iris dataset with config: %s', config)
    process_data(config)

if __name__ == "__main__":
    main()
