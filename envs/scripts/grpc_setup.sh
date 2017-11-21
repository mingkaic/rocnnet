#!/usr/bin/env bash

set -e

VERSION=${1:-1.7.1};

pushd /tmp

git clone -b v$VERSION https://github.com/grpc/grpc;

pushd grpc
git submodule update --init;
make && make install;
popd

popd
