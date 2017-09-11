#!/usr/bin/env bash

mkdir -pv /tmp/boostinst
cd /tmp/boostinst/
wget -c 'https://sourceforge.net/projects/boost/files/boost/1.64.0/boost_1_64_0.tar.bz2/download'
tar -xf download
cd boost_1_64_0
./bootstrap.sh --with-libraries=python
./bjam install
