#!/bin/bash
# https://doc.sagemath.org/html/en/installation/conda.html#sec-installation-conda

# Utility functions for colorcoding messages.
function info() {
    echo -e "\033[1;32mINFO:\033[0m $1"
}
function error() {
    echo -e "\033[1;31mERROR:\033[0m $1"
}
function warning() {
    echo -e "\033[1;33mWARNING:\033[0m $1"
}

# Check if conda is installed
if ! [ -x "$(command -v conda)" ]; then
    error "Conda is not installed. Please install conda and try again."
    exit 1
fi

# Install in virtual environment
conda config --add channels conda-forge
conda create -n sage-env python=3.11 -y
conda activate sage-env
conda install -c conda-forge sage -y
pip install --upgrade pip
pip install -r requirements.txt

# Verify that sage and jupyter are installed
if ! [ -x "$(command -v sage)" ]; then
    error "Sage is not installed. Please install sage and try again."
    exit 1
elif ! [ -x "$(command -v jupyter)" ]; then
    error "Jupyter is not installed. Please install jupyter and try again."
    exit 1
fi

# Install the kernel
sage -sh -c "ls -d $SAGE_VENV/share/jupyter/kernels/sagemath"
python -m ipykernel install --user --name=sage-env

# Add the kernel to Jupyter
jupyter kernelspec install --user "$(sage -sh -c "ls -d $SAGE_VENV/share/jupyter/kernels/sagemath")" --name sagemath

# Or start Jupyter notebook with sage kernel
# sage -n jupyter
# Troubleshooting: https://doc.sagemath.org/html/en/installation/launching.html