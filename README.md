# awesome_machine_learning
Implement various kind of Machine Learning Algorithm.
The idea and suggest code are come from blog:
https://machinelearningcoban.com/

Thankfully for the author: @Vu Huu Tiep. He is my star!

Each dir is a one specific machine learning topic or algorithm.

## Quick Start

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

Recommendation System (movielens): https://grouplens.org/datasets/movielens/100k/

### Run Recommendation System

Content based recommendation system (Make sure you download the dataset to data dir before running program)::

```bash
PYTHONPATH=. ./bin/content_based_rs
```

Neighboor based collaborative filtering recommendation system:

```bash
PYTHONPATH=. ./bin/neghboor_based_rs
```

### Run K Nearest Neighbors
Compare between regular and fast way when using simple knn

```bash
PYTHONPATH=. ./bin/k_nearest_neighbors
```