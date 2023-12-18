#!/usr/bin/env bash

conda create -n bert_hw python=3.7
conda activate bert_hw

conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.1 -c pytorch
pip install tqdm==4.58.0
pip install requests==2.25.1
pip install importlib-metadata==3.7.0
pip install filelock==3.0.12
pip install sklearn==0.0
pip install tokenizers
pip install transformers
pip install torch-optimizer
pip install --upgrade jax transformers
pip3 install emoji==0.6.0
pip install --upgrade jax jaxlib
pip install adabound

apt-get update
apt-get install -y build-essential
