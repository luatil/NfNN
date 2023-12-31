#!/bin/bash

opts="-Wall -Wno-unused-function -O3"
link_ops="-lm"
includes="lib"
out_dir="build"

# Ensure the output directory exists
mkdir -p "$out_dir"
# cd "$out_dir" || exit

echo "Build count lines"
gcc $opts -I"$includes" misc/count_lines.c -o "$out_dir"/count_lines $link_ops

echo "Build tests"
gcc $opts -I"$includes" tests/tests.c -o "$out_dir"/tests $link_ops

echo "Build examples - xor"
gcc $opts -I"$includes" examples/xor/mlp/src/xor.c -o "$out_dir"/xor $link_ops

echo "Build examples - mnist dataloader"
gcc $opts -I"$includes" examples/mnist/mnist_single_device/src/mnist_test_dataloader.c -o "$out_dir"/mnist_test_dataloader $link_ops

echo "Build examples - mnist single device"
gcc $opts -I"$includes" examples/mnist/mnist_single_device/src/mnist_single_device.c -o "$out_dir"/mnist_single_device $link_ops

echo "Build examples - mnist sync parameter server"
gcc $opts -I"$includes" examples/mnist/mnist_sync_parameter_server/src/mnist_sync_parameter_server.c -o "$out_dir"/mnist_sync_parameter_server $link_ops

echo "Build examples - mnist async parameter server"
gcc $opts -I"$includes" examples/mnist/mnist_async_parameter_server/src/mnist_async_parameter_server.c -o "$out_dir"/mnist_async_parameter_server $link_ops

pushd "$out_dir"
echo "All files" > ../misc/stats.txt
./count_lines .. >> ../misc/stats.txt
echo "Main lib" >> ../misc/stats.txt
./count_lines ../lib >> ../misc/stats.txt
echo "Examples" >> ../misc/stats.txt
./count_lines ../examples >> ../misc/stats.txt
popd

# cd ..

