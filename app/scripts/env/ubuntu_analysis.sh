#!/usr/bin/env bash
#
# purpose:
# this script installs all the analysis tools for ubuntu
#

set -e

GTEST_TAR=release-1.8.0.tar.gz;
GTEST_DIR=googletest-release-1.8.0;

VALG_TAR=valgrind-3.12.0.tar.bz2;
VALG_DIR=valgrind-3.12.0;

LCOV_TAR=lcov_1.13.orig.tar.gz;
LCOV_DIR=lcov-1.13;

# ===== Install Tests and Analysis Tools =====

# install gtest, gmock, valgrind sources
wget https://github.com/google/googletest/archive/$GTEST_TAR;
tar xf $GTEST_TAR;
pushd $GTEST_DIR;
cmake -DBUILD_SHARED_LIBS=ON . && make;
cp -a googlemock/include/gmock \
	googletest/include/gtest \
	/usr/include;
cp -a googlemock/libgmock_main.so \
	googlemock/libgmock.so \
	googlemock/gtest/libgtest_main.so \
	googlemock/gtest/libgtest.so /usr/lib/;
popd;
rm -rf $GTEST_DIR;
rm -rf $GTEST_TAR;

# install valgrind
wget http://valgrind.org/downloads/$VALG_TAR;
tar xvf $VALG_TAR;
pushd $VALG_DIR;
./configure --prefix=/usr && make;
make install;
popd;
rm -rf $VALG_DIR;
rm -rf $VALG_TAR;

# complete download gtest
# lacking libc6-dbg on 32-bit systems, use installer to install
apt-get -qq update;
apt-get install -y libgtest-dev libc6-dbg;

# download lcov for coverage analysis
wget http://ftp.de.debian.org/debian/pool/main/l/lcov/lcov_1.13.orig.tar.gz;
tar xf $LCOV_TAR;
make -C $LCOV_DIR/ install;
rm -rf $LCOV_DIR;
rm -rf $LCOV_TAR;

# download coverall-lcov (ruby)
gem install coveralls-lcov;

# download open-ai gym
if [ ! -e "gym" ];
then
	git clone https://github.com/openai/gym;
fi
pushd gym && pip install -e . && popd;
rm -rf gym

echo ""
echo "============ UBUNTU ANALYSIS TOOLS INSTALLED ============";
