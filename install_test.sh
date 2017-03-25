#!/bin/sh

# download lcov for coverage analysis
wget http://ftp.de.debian.org/debian/pool/main/l/lcov/lcov_1.13.orig.tar.gz
tar xf lcov_1.13.orig.tar.gz
sudo make -C lcov-1.13/ install

# download coverall-lconv (ruby)
gem install coveralls-lcov

# download valgrind for profiling
sudo apt-get -qq update
sudo apt-get install -y libgtest-dev valgrind

# download googletest (and gmock) and copy to /usr
sudo wget https://github.com/google/googletest/archive/release-1.8.0.tar.gz
sudo tar xf release-1.8.0.tar.gz
cd googletest-release-1.8.0
sudo cmake -DBUILD_SHARED_LIBS=ON .
sudo make
sudo cp -a googlemock/include/gmock /usr/include
sudo cp -a googletest/include/gtest /usr/include
sudo cp -a googlemock/libgmock_main.so googlemock/libgmock.so /usr/lib/
sudo cp -a googlemock/gtest/libgtest_main.so googlemock/gtest/libgtest.so /usr/lib/
