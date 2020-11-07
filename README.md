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

### Run Recommendation System

Content based recommendation systerm:

```bash
PYTHONPATH=. ./bin/content_based_rs
```

Neighboor based collaborative filtering recommendation system:

```bash
PYTHONPATH=. ./bin/neghboor_based_rs
```