from __future__ import print_function
import numpy as np
import pandas as pd

import settings

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Ridge
from sklearn import linear_model

def get_items_rated_by_user(rate_matrix, user_id):
    """Return scores of items for given user_id.
    """
    y = rate_matrix[:,0]

    # Get items rated by user_id
    ids = np.where(y == user_id + 1)[0]
    item_ids = rate_matrix[ids, 1] -1
    scores = rate_matrix[ids, 2]

    return (item_ids, scores)

def predict_for_user(rate_test, Yhat, user_id):
    """Return predict and true rating values of user.
    """
    np.set_printoptions(precision=2)
    ids, scores = get_items_rated_by_user(rate_test, user_id)
    print('Rated by movies ids: ', ids)
    print('True ratings: ', scores)
    print('Predict ratings: ', Yhat[ids, user_id])

def evaluate_model(num_users, Yhat, rates, W, b):
    """Evaluate for model.
    """
    se = cnt = 0
    for n in range(num_users):
        ids, scores_truth = get_items_rated_by_user(rates, n)
        scores_pred = Yhat[ids, n]
        e = scores_truth - scores_pred
        se += (e*e).sum(axis = 0)
        cnt += e.size

    return np.sqrt(se/cnt)

def process_data():
    # Read user file
    user_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('data/ml-100k/u.user', sep='|', names=user_cols)
    num_users = users.shape[0]
    print('Number of users: ', num_users)

    # Read rating file
    rate_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

    ratings_base = pd.read_csv('data/ml-100k/ua.base', sep='\t', names=rate_cols)
    ratings_test = pd.read_csv('data/ml-100k/ua.test', sep='\t', names=rate_cols)

    rate_train = ratings_base.to_numpy()
    print(rate_train)
    rate_test = ratings_test.to_numpy()

    print('Number of training rates: ', rate_train.shape[0])
    print('Number of testing rates: ', rate_test.shape[0])

    # Read items file
    item_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
                 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    items = pd.read_csv('data/ml-100k/u.item', sep='|', names=item_cols, encoding='latin-1')
    items_train = items.to_numpy()[:, -19:]
    print(items_train)
    print('Number of items: ', items_train.shape[0])

    # Build feature vectors for items
    transformer = TfidfTransformer(smooth_idf=True, norm='l2')
    X_train = transformer.fit_transform(items_train.tolist()).toarray()
    print(X_train)

    # Build Ridge Regression model
    d = X_train.shape[1]
    W = np.zeros((d, num_users))
    b = np.zeros(num_users)

    for n in range(num_users):
        ids, scores = get_items_rated_by_user(rate_train, n)
        model = Ridge(alpha=0.01, fit_intercept=True)
        Xhat = X_train[ids, :]
        model.fit(Xhat, scores)
        W[:, n] = model.coef_
        b[n] = model.intercept_

    # Predict scores
    Yhat = X_train.dot(W) + b

    # Example for user 100
    predict_for_user(rate_test, Yhat, 100)

    # Evaluate model
    rmse_train = evaluate_model(num_users, Yhat, rate_train, W, b)
    rmse_test = evaluate_model(num_users, Yhat, rate_test, W, b)
    print("RMSE for training phase: {}".format(rmse_train))
    print("RMSE for testing phase: {}".format(rmse_test))

if __name__ == "__main__":
    """Main program for content_based rec program.
    """
    process_data()