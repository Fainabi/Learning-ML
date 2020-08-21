#! /bin/bash
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P data/mnist
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P data/mnist
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P data/mnist
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P data/mnist

gzip -d data/mnist/train-images-idx3-ubyte.gz
gzip -d data/mnist/train-labels-idx1-ubyte.gz
gzip -d data/mnist/t10k-images-idx3-ubyte.gz
gzip -d data/mnist/t10k-labels-idx1-ubyte.gz
