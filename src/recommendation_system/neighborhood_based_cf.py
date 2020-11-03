from __future__ import print_function
import numpy as np
import pandas as pd
import logging

import settings
from src.config.config import read_config_file

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Ridge
from sklearn import linear_model

def process_data():
    pass

def main():
    """Main program for content_based rec program.
    """
    config = read_config_file(settings.config_file)
    if config['debug']:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug('Start neighborhood based RS with config: %s', config)
    process_data()

if __name__ == "__main__":
    main()