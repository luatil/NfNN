#!/bin/bash

echo "[LOG] Downloading MNIST"
pushd examples/mnist
./download_mnist.sh
popd
