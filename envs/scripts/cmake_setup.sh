#!/usr/bin/env bash

set -e

VERSION=${1:-3.8.2};
MAJOR=`expr "$VERSION" : '^\(.\d*\..\d*\)'`;
FOLDER=cmake-$VERSION;

pushd /tmp
apt-get purge cmake
wget http://www.cmake.org/files/v$MAJOR/$FOLDER.tar.gz;
tar xzf $FOLDER.tar.gz;
pushd $FOLDER
./bootstrap
make -j4 && make install;
popd
popd
