#!/bin/sh

# Add venv to jupyter
. ./.venv/bin/activate
pip install pip ipykernel -U
python -m ipykernel install --user --name=class_d_amp
deactivate

# Use symengine backend
export USE_SYMENGINE=1

# Start jupyter
jupyter-lab

