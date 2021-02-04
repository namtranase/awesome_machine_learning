# awesome_machine_learning
Implement various kind of Machine Learning Algorithm.
The idea and suggest code are come from blog:
(https://machinelearningcoban.com/)

Thankfully for the author of the book `Machine Learning co ban`: @Vu Huu Tiep.

Each dir is a one specific machine learning topic or algorithm.
## Project structure

```bash
├── bin
├── data
├── src
│   ├── config
│   │   ├──config.py
│   ├── k_means_clustering
│   ├── k_means_neighbors
│   ├── k_means_clustering
├── .gitignore
├── config.yaml
├── README.md
├── requirements.txt
├── settings.py
```
## Quick Start 🔥

### Setup Project

Clone source code from github:

```bash
git clone https://github.com/namtranase/awesome_machine_learning.git
cd awesome_machine_learning
```

Create virtual environment to install dependencies for project:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Link dataset

Recommendation System (movielens): (https://grouplens.org/datasets/movielens/100k/)

## Run Recommendation System 🔥

`Content based` recommendation system (Make sure you download the dataset to data dir before running program)::

```bash
PYTHONPATH=. ./bin/content_based_rs
```

`Neighboor based collaborative filtering` recommendation system:

```bash
PYTHONPATH=. ./bin/neghboor_based_rs
```

## Run K Nearest Neighbors 🔥

Compare between regular and fast way when using simple knn

```bash
PYTHONPATH=. ./bin/k_nearest_neighbors
```

Run knn for iris dataset, tuning to get best parameters

```bash
PYTHONPATH=. ./bin/k_nearest_neighbors_iris
```

## Run K Means Clustering 🔥

Compare simple kmeans and sklearn kmeans

```bash
PYTHONPATH=. ./bin/k_means_clustering
```

Run kmeans on mnist dataset

```bash
PYTHONPATH=. ./bin/k_means_clustering_mnist
```

### Run Naive Bayes Classifiers

Simple naive bayes based on sklearn

```bash
PYTHONPATH=. ./bin/naive_bayes_sklearn
```
