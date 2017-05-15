#!/usr/bin/env bash

PBX_CACHE=${1:-./prototxt}
ERRORLOG=test_out.txt
FUZZLOG=fuzz.out
BUILDDIR=.

on_travis() {
    if [ "$TRAVIS" == "true" ]; then
        "$@"
    fi
}

assert_cmd() {
    eval $*
    if [ $? -ne 0 ]; then
        echo "Command $* failed"
        cat ${ERRORLOG}
        cat ${FUZZLOG}
        exit 1;
    fi
    return $!
}

if [ "$TRAVIS" == "true" ]; then
    BUILDDIR=${TRAVIS_BUILD_DIR}
fi

if [ ! -d $PBX_CACHE ]; then
    mkdir $PBX_CACHE
fi

# compilation
pushd ${BUILDDIR}
lcov --directory . --zerocounters
mkdir build && pushd build
cmake -DTENNCOR_TEST=ON ..
cmake --build .

BINDIR=${BUILDDIR}/bin/bin

# valgrind check
for _ in {1..5}
do
    echo "running tenncortest with memcheck"
    assert_cmd "valgrind --tool=memcheck ${BINDIR}/tenncortest --gtest_shuffle > ${ERRORLOG}"
    echo "tenncortest valgrind check complete"
done

# tenncor tests
for _ in {1..5}
do
    echo "running batch of 5 tenncortest"
    assert_cmd "${BINDIR}/tenncortest --gtest_break_on_failure --gtest_repeat=5 --gtest_shuffle > ${ERRORLOG}"
    echo "still running! 5 tests complete"
done

# rocnnet demos
echo "running gd_demo"
assert_cmd "${BINDIR}/gd_demo $PBX_CACHE"
echo "gd_demo complete"

echo "running C++ dq_demo"
assert_cmd "${BINDIR}/dq_demo $PBX_CACHE"
echo "C++ dq_demo complete"

echo "running python dq_demo"
assert_cmd "python ${BUILDDIR}/demo/dq_demo.py"

# coverage analysis
pushd ${BUILDDIR}
lcov --version
gcov --version
lcov --directory . --gcov-tool gcov-6 --capture --output-file coverage.info # capture coverage info
lcov --remove coverage.info '**/gtest*' '**/tests/*' '**/build/*' '/usr/*' --output-file coverage.info # filter out system and test code
lcov --list coverage.info # debug < see coverage here
on_travis coveralls-lcov --repo-token ${COVERALLS_TOKEN} coverage.info # uploads to coveralls
