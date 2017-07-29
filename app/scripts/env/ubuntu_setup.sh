#!/bin/bash
#
# purpose:
# this script installs all the dependencies necessary to run the project on ubuntu
#

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $THIS_DIR/utils.sh

# ===== Define Installation Variables =====
set +e

REQ_GCC_VER="6.0.0";
CUR_GCC_VER="$( gcc -dumpversion )";

REQ_GPP_VER="6.0.0";
CUR_GPP_VER="$( g++ -dumpversion )";

REQ_CMAKE_VER="3.2.0";
CUR_CMAKE_VER="$( cmake --version )";

set -e

PB_TAR=protobuf-cpp-3.2.0.tar.gz;
PB_DIR=protobuf-3.2.0;

# ===== Apt-gettable Dependencies =====

apt-get update;
apt-get -qq update
apt-get autoremove -y;
apt-get clean;
apt-get -y dist-upgrade;
apt-get install -y \
    software-properties-common \
    python-software-properties \
    python-setuptools \
    build-essential \
    python-dev \
    python-tk \
    zip unzip \
    apt-utils \
    rubygems \
    swig \
    curl \
    wget \
    git;
add-apt-repository -y ppa:ubuntu-toolchain-r/test;
add-apt-repository -y ppa:george-edison55/cmake-3.x;
apt-get update;

# ===== Compilers and Managers =====

meets_version "$REQ_GCC_VER" "$CUR_GCC_VER" "apt-get install -y gcc-6 && rm /usr/bin/gcc && mv /usr/bin/gcc-6 /usr/bin/gcc";
meets_version "$REQ_GPP_VER" "$CUR_GPP_VER" "apt-get install -y g++-6 && rm /usr/bin/g++ && mv /usr/bin/g++-6 /usr/bin/g++";
meets_version "$REQ_CMAKE_VER" "$CUR_CMAKE_VER" "apt-get install -y cmake && apt-get -y upgrade";

# install pip
easy_install pip;

# ===== Other Dependencies =====

# install protobuf3
wget https://github.com/google/protobuf/releases/download/v3.2.0/$PB_TAR;
tar xvf $PB_TAR && pushd $PB_DIR && ./configure --prefix=/usr && make && make install && ldconfig && popd;
# clean up
rm -rf $PB_DIR
rm -rf $PB_TAR

# install pip requirements
pip install -r requirements.txt;

echo ""
echo "============ UBUNTU SETUP SUCCESS ============";