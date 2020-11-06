from __future__ import print_function
import logging

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

import settings
from src.config.config import read_config_file

class uuCF(object):
    """Class implement for user user collaborative filtering.
    """
    def __init__(self,
                 Y_data,
                 k,
                 sim_func=cosine_similarity):
        self.Y_data = Y_data
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

        for n in range(self.n_users):
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
        """Predict the rating of user u for item i.
        """
        # Find item i
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        # All users who rated i
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        # Similarity of u and users who rated i
        sim = self.S[u, users_rated_i]
        # Most k similar users
        nns = np.argsort(sim)[-self.k:]
        nearest_s = sim[nns]
        # The correctsponding ratings
        r = self.Ybar[i, users_rated_i[nns]]
        eps = 1e-8 # Avoid zero division
        return (r*nearest_s).sum()/(np.abs(nearest_s).sum() + eps) + self.mu[u]

def process_data():
    """Process neighboor_based_cf program.
    """
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings_base = pd.read_csv('data/ml-100k/ua.base', sep='\t', names=r_cols)
    ratings_test = pd.read_csv('data/ml-100k/ua.test', sep='\t', names=r_cols)

    rate_train = ratings_base.to_numpy()
    rate_test = ratings_test.to_numpy()

    # Indices start form 0
    rate_train[:, :2] -= 1
    rate_test[:, :2] -= 1

    rs = uuCF(rate_train, k = 40)
    rs.fit()

    n_tests = rate_test.shape[0]
    SE = 0 # squared error
    for n in range(n_tests):
        pred = rs.pred(rate_test[n, 0], rate_test[n, 1])
        SE += (pred - rate_test[n,2])**2

    RMSE = np.sqrt(SE/n_tests)
    logging.debug('Uses-user CF, RMSE: %s', RMSE)
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
