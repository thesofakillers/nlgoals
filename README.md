# Natural Language for Goal Misgeneralization

## Requirements and Setup

Details such as python and package versions can be found in the generated
[pyproject.toml](pyproject.toml) and [poetry.lock](poetry.lock) files.

We recommend using an environment manager such as
[conda](https://docs.conda.io/en/latest/). After setting up your environment
with the correct python version, please proceed with the installation of the
required packages

First, install the pre-requirements[^1]:

```terminal
pip install -r requirements/pre-reqs.txt
```

For [poetry](https://python-poetry.org/) users, getting setup is then as easy as
running

```terminal
poetry install
```

We also provide `requirements.txt` files for
[pip](https://pypi.org/project/pip/) users who do not wish to use poetry. We
provide a [requirements-minimal.txt](requirements/requirements-minimal.txt) file
and a [requirements-complete.txt](requirements/requirements-complete.txt) file
in the [requirements](requirements/) directory. We recommend simply running

```terminal
pip install -r requirements/requirements-complete.txt
```

These `requirements.txt` file are generated by running
[gen_pip_reqs.sh](gen_pip_reqs.sh)

[^1]:
    This annoying first step is necessary because of stupid packages we depend
    on like pyhash and multicoretsne that don't know how to get their
    dependencies sorted out like everyone else. See
    [this issue](https://github.com/DmitryUlyanov/Multicore-TSNE/issues/81#issuecomment-863745998)
    and [this issue](https://github.com/flier/pyfasthash/issues/59). These two
    packages are the main culprits but other packages haven't been very good at
    defining their dependencies either.

## Project Organization

```plaintext
    ├── LICENSE
    ├── README.md          <- The top-level README
    ├── data/              <- Datasets
    ├── checkpoints/       <- Trained and serialized models.
    ├── notebooks/         <- Jupyter notebooks.
    ├── documents/         <- Documents as HTML, PDF, LaTeX, etc.
    ├── pyproject.toml     <- Project metadata, handled by poetry.
    ├── poetry.lock        <- Resolving and locking dependencies, handled by poetry.
    ├── requirements.txt   <- For non-poetry users.
    ├── gen_pip_reqs.sh    <- For generating the pip requirements.txt file
    ├── tests/             <- Tests
    ├── outputs/           <- Output files. Not committed.
    └── src/nlgoals/       <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        ├── data/          <- Scripts to download or generate data
        ├── models/        <- Model definitions
        ├── run/           <- Scripts to train, evaluate and use models
        └── visualization/ <- Scripts for visualization
```

The project structure is largely based on the
[cookiecutter data-science template](https://github.com/drivendata/cookiecutter-data-science).
This is purposely opinionated so that paths align over collaborators without
having to edit config files. Users may find the
[cookiecutter data-science opinions page](http://drivendata.github.io/cookiecutter-data-science/#opinions),
of relevance

The top level `data/` and `models/` directory are in version control only to
show structure. Their contents will not be committed and are ignored via
`.gitignore`.

---
