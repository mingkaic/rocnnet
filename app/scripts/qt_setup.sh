#!/usr/bin/env bash

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";
pushd $THIS_DIR/..
git submodule update --init
popd

apt install qtbase5-dev
