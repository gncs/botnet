#!/usr/bin/env bash
CUDA="cpu"

# Create environment and activate
# python3.9 -m venv e3nn-ff-venv
# source e3nn-ff-venv/bin/activate

pip install pip --upgrade
pip install wheel pytest pyyaml mypy flake8 yapf

pip install torch==1.8.1
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.htmlr

pip install sympy
pip install opt_einsum_fx==0.1.1
pip install torch-geometric==1.7.0
pip install e3nn==0.3.2
