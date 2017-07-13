#!/bin/bash
#
# purpose:
# this script installs all the dependencies necessary to run the project
#

function usage() {
    echo ""
    echo "$0: <unpack location> <build directory> <installer command>";
    echo "OPTIONAL:";
    echo "  Unpack location is used to store downloaded dependent packages";
    echo "  Build directory is the base folder";
    echo "  Installer command is any support installer";
    exit 1;
}

# ===== Check and define directories

CMAKEDIR="cmake-3.2.2";
PROTODIR="protobuf-3.2.0";
GTESTDIR="googletest-release-1.8.0";

CACHEDIR="rocnnet-dep";
BUILDDIR=".";
INSTALLER="apt-get";

# ===== Parameter sanity check =====

if [ ! -z $1 ];
then
    CACHEDIR=$1;
fi

if [ ! -z $2 ];
then
    BUILDDIR=$2;
fi

if [ ! -z $3 ];
then
    INSTALLER=$3;
fi

if [ ! -d $CACHEDIR ];
then
    mkdir $CACHEDIR;
fi

if [ ! -d $BUILDDIR ];
then
    echo "Build directory $BUILDDIR does not exist";
    usage;
fi

# ===== Define functions

# executes command CMD, echo usage if it fails
function exec_cmd() {
    CMD=$*;
    eval $CMD;
    if [ $? -ne 0 ];
    then
        echo "Command $CMD failed";
        usage;
    fi
    return $!
}

# executes command CMD if required REQ_VER is greater than CUR_VER
function meets_version() {
    REQ_VER=$1;
    CUR_VER=$2;
    CMD=${@:3:$#-2};
    echo "YOOO $CMD"
    if [ "$(printf "$REQ_VER\n$CUR_VER" | sort -V | head -n1)" == "$CUR_VER" ] && [ "$CUR_VER" != "$REQ_VER" ];
    then
        exec_cmd "$CMD";
    else
        echo "requirement: $REQ_VER met. current version: $CUR_VER";
    fi
}

# executes installation command CMD using global INSTALLER
function update_install() {
    CMD=$*;
    $INSTALLER update && $INSTALLER install -y $CMD;
}

# download from SRC_LINK, uncompress TAR_NAME, configure and build in TAR_DIR
function download_cfg_build() {
    SRC_LINK=$1;
    TAR_NAME=$2;
    TAR_DIR=$3;
    wget $SRC_LINK && tar xvfz $TAR_NAME && pushd $TAR_DIR && ./configure --prefix=/usr && make && popd;
}

# execute command CMD in cache and execute a command only if TARDIR directory does not exist
function incache() {
    TARDIR=$1;
    CMD=${@:2:$#-1};
    if [ ! -d $CACHEDIR/$TARDIR/ ];
    then
        echo "Not Found: $CACHEDIR/$TARDIR/";
        exec_cmd "pushd $CACHEDIR && $CMD && popd";
    fi
}

# ===== Grab apt-get repositories =====

if [ "$INSTALLER" == "apt-get" ];
then
    exec_cmd "apt-get autoremove -y";
    exec_cmd "apt-add-repository -y ppa:ubuntu-toolchain-r/test";
fi

# ===== Install ruby, python and essentials =====

REQ_GEM_VER="1.9.1";
CUR_GEM_VER="$(gcc -v)";
meets_version $REQ_GEM_VER $CUR_GEM_VER "update_install ruby 1.9.3";

# install pip
update_install python-setuptools python-dev build-essential;
easy_install pip;

# ===== Install/Upgrade compilers =====

REQ_GCC_VER="6.0.0";
CUR_GCC_VER="$(gcc -dumpversion)";
meets_version $REQ_GCC_VER $CUR_GCC_VER "update_install gcc-6 && rm /usr/bin/gcc && mv /usr/bin/gcc-6 /usr/bin/gcc";

REQ_GPP_VER="6.0.0";
CUR_GPP_VER="$(g++ -dumpversion)";
meets_version $REQ_GPP_VER $CUR_GPP_VER "update_install g++-6 && rm /usr/bin/g++ && mv /usr/bin/g++-6 /usr/bin/g++";

# ===== Install Cmake =====

REQ_CMAKE_VER="3.0.0";
CUR_CMAKE_VER="$(cmake --version)";
meets_version $REQ_CMAKE_VER $CUR_CMAKE_VER "update_install cmake";

# ===== Build dependencies =====

# download protobuf3 and cache if necessary
incache $PROTODIR "download_cfg_build https://github.com/google/protobuf/releases/download/v3.2.0/protobuf-cpp-3.2.0.tar.gz protobuf-cpp-3.2.0.tar.gz $PROTODIR";
exec_cmd "pushd $CACHEDIR/$PROTODIR && make install && ldconfig && popd"

# install swig
exec_cmd "update_install swig";

# install PythonLibs and tk
exec_cmd "update_install python-dev python-tk";

# install python requirements
exec_cmd "pushd $BUILDDIR && pip install -r requirements.txt && popd";

update_install checkinstall;
checkinstall -y;

# ===== Install tests and analysis tools =====

# download googletest (and gmock) and cache if necessary
incache $GTESTDIR "wget https://github.com/google/googletest/archive/release-1.8.0.tar.gz && tar xf release-1.8.0.tar.gz"
if [ ! -e "$CACHEDIR/$GTESTDIR/googlemock/libgmock_main.so" ];
then
    exec_cmd "pushd $CACHEDIR/$GTESTDIR && cmake -DBUILD_SHARED_LIBS=ON . && make && popd";
fi
# move googletest to /usr
exec_cmd "pushd $CACHEDIR/$GTESTDIR && cp -a googlemock/include/gmock googletest/include/gtest /usr/include && popd";
exec_cmd "pushd $CACHEDIR/$GTESTDIR && cp -a googlemock/libgmock_main.so googlemock/libgmock.so googlemock/gtest/libgtest_main.so googlemock/gtest/libgtest.so /usr/lib/ && popd";

# download valgrind for profiling
update_install libgtest-dev valgrind;

# download lcov for coverage analysis
wget http://ftp.de.debian.org/debian/pool/main/l/lcov/lcov_1.13.orig.tar.gz;
tar xf lcov_1.13.orig.tar.gz;
make -C lcov-1.13/ install;

# download coverall-lconv (ruby)
gem install coveralls-lcov;

# download open-ai gym
if [ ! -e "$BUILDDIR/gym" ];
then
    exec_cmd "pushd $BUILDDIR/ && git clone https://github.com/openai/gym && popd";
fi
exec_cmd "pushd $BUILDDIR/gym && pip install -e . && popd";
echo ""
echo "============ SETUP SUCCESS============";