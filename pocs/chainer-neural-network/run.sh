#!/bin/bash

wget https://github.com/chainer/chainer/archive/v7.8.1.tar.gz
tar xzf v7.8.1.tar.gz
python3 chainer-7.8.1/examples/mnist/train_mnist.py
