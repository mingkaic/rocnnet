#!/bin/bash
#
# purpose:
# this script installs all the dependencies necessary to run the project
#

function usage() {
    echo "$0: <unpack location> <build directory>";
    echo "  Unpack location is used to store downloaded dependent packages";
    echo "  Build directory is the base folder";
    exit 1;
}

if [ $# -ne 2 ]; then
    usage
fi

if [ ! -d $1 ]; then
    echo "Unpack directory $1 does not exist"
    echo ""
    usage;
fi

if [ ! -d $2 ]; then
    echo "Build directory $2 does not exist"
    echo ""
    usage;
fi

exec_cmd() {
    eval $*
    if [ $? -ne 0 ]; then
        echo "Command $* failed"
        usage
    fi
    return $!
}

PROTODIR="protobuf-3.2.0"
GTESTDIR="googletest-release-1.8.0"

# ===== install installation dependencies =====
sudo apt-get install -y ruby1.9.3
gem1.9.1 --version

# ===== build dependencies =====

# install compilers
exec_cmd "apt-get autoremove -y"
exec_cmd "apt-add-repository -y ppa:ubuntu-toolchain-r/test"
exec_cmd "apt-get update && apt-get install -y g++-6 gcc-6"
rm /usr/bin/g++ 
rm /usr/bin/gcc
exec_cmd "mv /usr/bin/g++-6 /usr/bin/g++"
exec_cmd "mv /usr/bin/gcc-6 /usr/bin/gcc"

# download protobuf3 and cache if necessary
if [ ! -d $1/$PROTODIR/ ]; then
    echo "Not Found: $1/$PROTODIR/"
    exec_cmd "pushd $1/ && wget https://github.com/google/protobuf/releases/download/v3.2.0/protobuf-cpp-3.2.0.tar.gz && popd";
    exec_cmd "pushd $1/ && tar xvfz protobuf-cpp-3.2.0.tar.gz && pushd $PROTODIR && ./configure --prefix=/usr && make && popd && popd";
fi
# install protobuf3
exec_cmd "pushd $1/$PROTODIR && make install && ldconfig && popd"

# install swig
exec_cmd "apt-get update && apt-get install -y swig"

# install PythonLibs and tk
exec_cmd "apt-get update && apt-get install -y python-dev python-tk"

# install python requirements
exec_cmd "pushd $2 && pip install -r requirements.txt && popd"

# ===== tests and analysis tools =====

# download googletest (and gmock) and cache if necessary
if [ ! -d $1/$GTESTDIR/ ]; then
    echo "Not Found: $1/$GTESTDIR/"
    exec_cmd "pushd $1/ && wget https://github.com/google/googletest/archive/release-1.8.0.tar.gz && popd"
    exec_cmd "pushd $1/ && tar xf release-1.8.0.tar.gz && pushd $GTESTDIR && cmake -DBUILD_SHARED_LIBS=ON . && make && popd && popd"
fi
# move googletest to /usr
exec_cmd "pushd $1/$GTESTDIR && cp -a googlemock/include/gmock googletest/include/gtest /usr/include && popd"
exec_cmd "pushd $1/$GTESTDIR && cp -a googlemock/libgmock_main.so googlemock/libgmock.so googlemock/gtest/libgtest_main.so googlemock/gtest/libgtest.so  /usr/lib/ && popd"

# download valgrind for profiling
apt-get -qq update
apt-get install -y libgtest-dev valgrind

# download lcov for coverage analysis
wget http://ftp.de.debian.org/debian/pool/main/l/lcov/lcov_1.13.orig.tar.gz
tar xf lcov_1.13.orig.tar.gz
make -C lcov-1.13/ install

# download coverall-lconv (ruby)
gem1.9.1 install coveralls-lcov

# downlaod open-ai gym
exec_cmd "pushd $2/ && git clone https://github.com/openai/gym && pushd gym && pip install -e . && popd"
