#!/usr/bin/env bash

on_travis() {
    if [ "$TRAVIS" == "true" ]; then
        "$@"
    fi
}

assert_cmd() {
    eval $*
    if [ $? -ne 0 ]; then
        echo "Command $* failed"
        exit 1;
    fi
    return $!
}

PBX_CACHE=${1:./prototxts}

if [ ! -d $PBX_CACHE ]; then
    mkdir $PBX_CACHE
fi

# compilation
on_travis pushd ${TRAVIS_BUILD_DIR}
lcov --directory . --zerocounters
mkdir build && pushd build
cmake -DTENNCOR_TEST=ON -DCMAKE_CXX_COMPILER=g++-6 ..
cmake --build .

# tenncor tests
on_travis assert_cmd "${TRAVIS_BUILD_DIR}/tenncor/bin/bin/tenncortest --gtest_break_on_failure --gtest_repeat=50 --gtest_shuffle && cat fuzz.out"

# rocnnet demos
on_travis assert_cmd "${TRAVIS_BUILD_DIR}/bin/bin/gd_demo $PBX_CACHE"

on_travis pushd ${TRAVIS_BUILD_DIR}
lcov --version
gcov --version
lcov --directory . --gcov-tool gcov-6 --capture --output-file coverage.info # capture coverage info
lcov --remove coverage.info '**/gtest*' '**/tests/*' '/usr/*' --output-file coverage.info # filter out system and test code
lcov --list coverage.info # debug < see coverage here
on_travis coveralls-lcov --repo-token ${COVERALLS_TOKEN} coverage.info # uploads to coveralls
