#!/bin/sh

# Activate virtualenv
source .venv/bin/activate

# Install dependencies
pip install -U pip
pip install -U ipykernel
pip install -U jupyterlab

# Add venv to jupyter
ipython kernel install --user --name=.venv

# Start jupyter
jupyter lab

