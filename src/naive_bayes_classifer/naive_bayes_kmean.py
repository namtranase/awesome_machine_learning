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
    means = [[2, 2], [8, 3], [3, 6]]
    cov = [[1, 0], [0, 1]]
    N = 500
    X0 = np.random.multivariate_normal(means[0], cov, N)
    X1 = np.random.multivariate_normal(means[1], cov, N)
    X2 = np.random.multivariate_normal(means[2], cov, N)

    X = np.concatenate((X0, X1, X2), axis=0)

    original_labels = np.asarray([0]*N + [1]*N + [2]*N).T

    return X, original_labels

def kmeans_init_centroids(X, k):
    """Randomly choose k rows of X as centroids.
    """
    return X[np.random.choice(X.shape[0], k, replace=False)]

def kmeans_assign_labels(X, centroids):
    """Calculate pairwise distances btw data and centroids.
    """
    D = cdist(X, centroids)

    return np.argmin(D, axis=1)

def has_converged(centroids, new_centroids):
    """Check the new centroid is whether smaller
    or bigger than the old one.
    """
    return (set([tuple(a) for a in centroids]) ==
        set([tuple(b) for b in new_centroids]))

def kmeans_update_centroids(X, labels, K):
    """Update the new centroids for data.
    """
    centroids = np.zeros((K, X.shape[1]))
    for k in range (K):
        X_k = X[labels==k, :]
        centroids[k,:] = np.mean(X_k, axis=0)

    return centroids

def kmeans(X, K):
    """Implement kmeans algorithm in easy way.
    """
    centroids = [kmeans_init_centroids(X, K)]
    labels = []
    it = 0
    while True:
        logging.debug('Iteration: %s', it)
        labels.append(kmeans_assign_labels(X, centroids[-1]))
        new_centroids = kmeans_update_centroids(X, labels[-1], K)
        if has_converged(centroids[-1], new_centroids):
            break
        centroids.append(new_centroids)
        it += 1

    return (centroids, labels, it)

def process_data():
    """Process KNN program.
    """
    np.random.seed(18)
    # Define 3 clusters
    K = 3

    # Prepare data
    X, original_labels = prepare_data()
    logging.debug('Numbers of data: %s', len(X))
    logging.debug('Labels of data: %s', set(original_labels))

    # Simple kmeans
    (centroids, labels, it) = kmeans(X, K)
    logging.debug('Centrel found by simple kmeans: %s', centroids[-1])

    # scikit-learn kmeans
    model = KMeans(n_clusters=3, random_state=0).fit(X)
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
