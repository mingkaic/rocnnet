#!/bin/bash
#
# purpose:
# this script installs all the dependencies necessary to run the project
#

# ===== Check and define directories

PROTODIR="protobuf-3.2.0"
GTESTDIR="googletest-release-1.8.0"
INSTALLER="apt-get"

# ===== Parameter sanity check =====

if [ $# -ne 2 ];
then
    usage
fi

if [ ! -d $1 ];
then
    echo "Unpack directory $1 does not exist"
    echo ""
    usage;
fi

if [ ! -d $2 ];
then
    echo "Build directory $2 does not exist"
    echo ""
    usage;
fi

if [ ! -z $3 ];
then
    INSTALLER=$3
fi

# ===== Define functions

function usage() {
    echo "$0: <unpack location> <build directory>";
    echo "  Unpack location is used to store downloaded dependent packages";
    echo "  Build directory is the base folder";
    exit 1;
}

# executes command, echo usage if it fails
function exec_cmd() {
    eval $*
    if [ $? -ne 0 ];
    then
        echo "Command $* failed"
        usage
    fi
    return $!
}

# executes commands (third param) if required version (first param) is greater than current version (second param)
function meets_version() {
    REQ_VER=#1
    CUR_VER=#2
    EXC_CMD=#3
    if [ "$(printf "$REQ_VER\n$CUR_VER" | sort -V | head -n1)" == "$CUR_VER" ] && [ "$CUR_VER" != "$REQ_VER" ];
    then
        exec_cmd $EXC_CMD
    fi
}

# executes installation using INSTALLER
function install() {
    $INSTALLER update && $INSTALLER install -y $*
}

# ===== Grab apt-get repositories =====

if [ "$INSTALLER" == "apt-get" ];
then
    exec_cmd "apt-get autoremove -y"
    exec_cmd "apt-add-repository -y ppa:ubuntu-toolchain-r/test"
fi

# ===== Install ruby, python and essentials =====

REQ_GEM_VER="1.9.1"
CUR_GEM_VER="$(gcc -v)"
meets_version $REQ_GEM_VER $CUR_GEM_VER "install ruby 1.9.3"

# install pip
$INSTALLER install -y python-setuptools python-dev build-essential
easy_install pip

# ===== Install/Upgrade compilers =====

REQ_GCC_VER="6.0.0"
CUR_GCC_VER="$(gcc -dumpversion)"
meets_version $REQ_GCC_VER $CUR_GCC_VER "install gcc-6 && rm /usr/bin/gcc && mv /usr/bin/gcc-6 /usr/bin/gcc"

REQ_GPP_VER="6.0.0"
CUR_GPP_VER="$(g++ -dumpversion)"
meets_version $REQ_GPP_VER $CUR_GPP_VER "install g++-6 && rm /usr/bin/g++ && mv /usr/bin/g++-6 /usr/bin/g++"

# ===== Install Cmake =====

REQ_GPP_VER="3.0.0"
CUR_GPP_VER="$(cmake --version)"
meets_version $REQ_CMAKE_VER $CUR_CMAKE_VER "install build-essential && wget http://www.cmake.org/files/v3.2/cmake-3.2.2.tar.gz && tar xf cmake-3.2.2.tar.gz && pushd cmake-3.2.2 && ./configure && make && popd"

install checkinstall
checkinstall -y

# ===== Build dependencies =====

# download protobuf3 and cache if necessary
if [ ! -d $1/$PROTODIR/ ];
then
    echo "Not Found: $1/$PROTODIR/"
    exec_cmd "pushd $1/ && wget https://github.com/google/protobuf/releases/download/v3.2.0/protobuf-cpp-3.2.0.tar.gz && popd";
    exec_cmd "pushd $1/ && tar xvfz protobuf-cpp-3.2.0.tar.gz && pushd $PROTODIR && ./configure --prefix=/usr && make && popd && popd";
fi
# install protobuf3
exec_cmd "pushd $1/$PROTODIR && make install && ldconfig && popd"

# install swig
exec_cmd "install swig"

# install PythonLibs and tk
exec_cmd "install python-dev python-tk"

# install python requirements
exec_cmd "pushd $2 && pip install -r requirements.txt && popd"

# ===== Install tests and analysis tools =====

# download googletest (and gmock) and cache if necessary
if [ ! -d $1/$GTESTDIR/ ];
then
    echo "Not Found: $1/$GTESTDIR/"
    exec_cmd "pushd $1/ && wget https://github.com/google/googletest/archive/release-1.8.0.tar.gz && popd"
    exec_cmd "pushd $1/ && tar xf release-1.8.0.tar.gz && popd"
fi
if [ ! -e "googlemock/libgmock_main.so" ];
then
    exec_cmd "pushd $1/$GTESTDIR && cmake -DBUILD_SHARED_LIBS=ON . && make && popd"
fi
# move googletest to /usr
exec_cmd "pushd $1/$GTESTDIR && cp -a googlemock/include/gmock googletest/include/gtest /usr/include && popd"
exec_cmd "pushd $1/$GTESTDIR && cp -a googlemock/libgmock_main.so googlemock/libgmock.so googlemock/gtest/libgtest_main.so googlemock/gtest/libgtest.so /usr/lib/ && popd"

# download valgrind for profiling
install libgtest-dev valgrind

# download lcov for coverage analysis
wget http://ftp.de.debian.org/debian/pool/main/l/lcov/lcov_1.13.orig.tar.gz
tar xf lcov_1.13.orig.tar.gz
make -C lcov-1.13/ install

# download coverall-lconv (ruby)
gem install coveralls-lcov

# download open-ai gym
if [ ! -e "$2/gym" ];
then
    exec_cmd "pushd $2/ && git clone https://github.com/openai/gym && popd"
fi
exec_cmd "pushd $2/gym && pip install -e . && popd"
echo "============ SETUP SUCCESS============"