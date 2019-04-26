#!/bin/bash

sudo apt-get install -y cmake libgflags-dev libgoogle-glog-dev bc
git clone https://github.com/pyotr777/DNNMark.git
mkdir -p ~/cudnn
cd DNNMark
git checkout convolutions
./setup.sh CUDA
cd build
make


