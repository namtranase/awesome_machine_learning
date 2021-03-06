from __future__ import print_function
from sklearn.naive_bayes import MultinomialNB
import logging
from time import time

import numpy as np

import settings
from src.config.config import read_config_file

def prepare_data(config):
    """Prepare data for training.
    """
    # Train data
    d1 = [2, 1, 1, 0, 0, 0, 0, 0, 0]
    d2 = [1, 1, 0, 1, 1, 0, 0, 0, 0]
    d3 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
    d4 = [0, 1, 0, 0, 0, 0, 1, 1, 1]

    train_data = np.array([d1, d2, d3, d4])
    labels = np.array(['B', 'B', 'B', 'N'])
    logging.debug("Number of training samples: %s", len(train_data))

    return train_data, labels

def prepare_bernoulli_data(config):
    """Prepare data for training.
    """
    # Train data
    d1 = [2, 1, 1, 0, 0, 0, 0, 0, 0]
    d2 = [1, 1, 0, 1, 1, 0, 0, 0, 0]
    d3 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
    d4 = [0, 1, 0, 0, 0, 0, 1, 1, 1]

    train_data = np.array([d1, d2, d3, d4])
    labels = np.array(['B', 'B', 'B', 'N'])
    logging.debug("Number of training samples: %s", len(train_data))

    return train_data, labels

def process_data(config):
    """Process KNN program.
    """
    # Prepare train data and their lables
    train_data, labels = prepare_data(config)

    # Simple Naive bayes model
    model = MultinomialNB()
    model.fit(train_data, labels)

    # Test results
    d5 = np.array([[2, 0, 0, 1, 0, 0, 0, 1, 0]])
    d6 = np.array([[0, 1, 0, 0, 0, 0, 0, 1, 1]])
    logging.debug("Predict class of d5: %s", str(model.predict(d5)[0]))
    logging.debug("Predict prob of d6 in each class: %s", model.predict_proba(d6))

def main():
    """Main program for Naive bayes program.
    """
    config = read_config_file(settings.config_file)
    if config['debug']:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug('Start Naive bayes with config: %s', config)
    process_data(config)

if __name__ == "__main__":
    main()
