#!/usr/bin/env bash

set -e

pushd /tmp

wget 'https://sourceforge.net/projects/boost/files/boost/1.64.0/boost_1_64_0.tar.bz2/download';
tar -xf download;

pushd boost_1_64_0
./bootstrap.sh --with-libraries=python,program_options;
./bjam install;
ldconfig
popd

popd
