#!/bin/sh

# Add venv to jupyter
<<<<<<< Updated upstream
. ./.venv/bin/activate
=======
. .venv/bin/activate
>>>>>>> Stashed changes
pip install pip ipykernel -U
python -m ipykernel install --user --name=class_d_amp
deactivate

<<<<<<< Updated upstream
# Use symengine backend
=======
# Use SYMENGINE backend for SymPy
>>>>>>> Stashed changes
export USE_SYMENGINE=1

# Start jupyter
jupyter-lab

