#!/bin/bash

# Formatting all .h files
find . -iname "*.h" -exec clang-format -i --style=Microsoft {} \;

# Formatting all .c files
find . -iname "*.c" -exec clang-format -i --style=Microsoft {} \;
