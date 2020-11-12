from __future__ import print_function
import logging
from time import time

import numpy as np

import settings
from src.config.config import read_config_file

def process_data():
    """Process KNN program.
    """

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
