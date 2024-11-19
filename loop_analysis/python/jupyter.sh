#!/bin/sh

# Add venv to jupyter
source ./.venv/bin/activate
pip install pip ipykernel -U
python -m ipykernel install --user --name=class_d_amp
deactivate

# Start jupyter
jupyter-lab

