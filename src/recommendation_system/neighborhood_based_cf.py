from __future__ import print_function
import numpy as np
import pandas as pd
import logging

import settings
from src.config.config import read_config_file

from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

class uuCF(object):
    def __init__(self,
                 Y_data,
                 k,
                 sim_func=cosin_similarity):
        # Number of neighborhood
        self.k = k
        # Similarity function, default: cosin sim
        self.sim_func = sim_func
        self.Ybar = None
        # Number of users
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        # Number of items
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1
    
    def fit(self):
        # Normalized Y_data -> Ybar
        users = self.Y_data[:, 0]
        self.Ybar = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))

        for n in xrange(self.n_users):
            # Row indices of rating of user n
            ids = np.where(users == n)[0].astype(np.int32)
            # Indices of all items rated by user n
            item_ids = self.Y_data[ids, 1]
            # Ratings made by user n
            ratings = self.Y_data[ids, 2]
            # Avoid zero division
            self.mu[n] = np.mean(ratings) if ids.size > 0 else 0
            self.Ybar[ids, 2] = ratings - self.mu[n]

        # Form the rating matrix as a sparse matrix
        self.Ybar = sparse.coo_matrix(
            (self.Ybar[:, 2], (self.Ybar[:, 1], self.Ybar[:, 0])),
            (self.n_items, self.n_users)).tocsr()

        self.S = self.sim_func(self.Ybar.T, self.Ybar.T)

    def pred(self, u, i):
        pass

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