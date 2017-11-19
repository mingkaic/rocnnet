#!/usr/bin/env bash

set -e

VERSION=${1:-3.4.0};

pushd /tmp

wget https://github.com/google/protobuf/releases/download/v$VERSION/protobuf-cpp-$VERSION.tar.gz;
tar xvf protobuf-cpp-$VERSION.tar.gz;

pushd protobuf-$VERSION
./configure && make && make install;
set +e
ldconfig;
set -e
popd

popd
