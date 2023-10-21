#!/bin/bash

opts="-g -Wall -Wno-unused-function"
link_ops="-lm"
includes="lib"
out_dir="build"

# Ensure the output directory exists
mkdir -p "$out_dir"
# cd "$out_dir" || exit

# Build tests
gcc $opts -I"$includes" tests/tests.c -o "$out_dir"/tests $link_ops

# Build examples - xor
gcc $opts -I"$includes" examples/xor/mlp/src/xor.c -o "$out_dir"/xor $link_ops

# Build examples - mnist dataloader
gcc $opts -I"$includes" examples/mnist/mnist_single_device/src/mnist_test_dataloader.c -o "$out_dir"/mnist_test_dataloader $link_ops

# Build examples - mnist single device
gcc $opts -I"$includes" examples/mnist/mnist_single_device/src/mnist_single_device.c -o "$out_dir"/mnist_single_device $link_ops

# Build examples - mnist sync parameter server
gcc $opts -I"$includes" examples/mnist/mnist_sync_parameter_server/src/mnist_sync_parameter_server.c -o "$out_dir"/mnist_sync_parameter_server $link_ops

# Build examples - mnist async parameter server
gcc $opts -I"$includes" examples/mnist/mnist_async_parameter_server/src/mnist_async_parameter_server.c -o "$out_dir"/mnist_async_parameter_server $link_ops

# cd ..

