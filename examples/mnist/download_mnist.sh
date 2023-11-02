#!/bin/bash

# Development based on https://gist.github.com/dvsseed/ad0b5526ae461a29f5268a4d0afdbd83

# Download the mnist data
mkdir --verbose --parents dataset/MNIST/raw/
pushd dataset/MNIST/raw/

wget --no-check-certificate http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget --no-check-certificate http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget --no-check-certificate http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget --no-check-certificate http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gzip -f -d train-labels-idx1-ubyte.gz
gzip -f -d train-images-idx3-ubyte.gz
gzip -f -d t10k-labels-idx1-ubyte.gz
gzip -f -d t10k-images-idx3-ubyte.gz
popd
