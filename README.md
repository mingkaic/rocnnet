[![Build Status](https://travis-ci.org/mingkaic/rocnnet.svg?branch=master)](https://travis-ci.org/mingkaic/rocnnet)
[![Coverage Status](https://coveralls.io/repos/github/mingkaic/rocnnet/badge.svg?branch=master)](https://coveralls.io/github/mingkaic/rocnnet?branch=master)

## Synopsis

ROCNNet is a neural net implemented in C++ using custom library Tenncor.

## Build

CMake 3.6 is required.

Download cmake: https://cmake.org/download/

Building from another directory is recommended:

    mkdir build 
    cd build
    cmake <path/to/rocnnet>
    make

Binaries and libraries should be found in the /bin directory
