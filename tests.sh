#!/usr/bin/env bash

on_travis() {
	if [ "$TRAVIS" == "true" ];
	then
		"$@"
	fi
}

BUILDDIR=.
if [ "$TRAVIS" == "true" ];
then
	BUILDDIR=$TRAVIS_BUILD_DIR
fi

LOGDIR=$BUILDDIR/log
PBX_CACHE=${1:-./prototxt}
ERRORLOG=$LOGDIR/test_out.txt
FUZZLOG=fuzz.out
TIMEOUT=600

assert_cmd() {
	eval timeout -s SIGKILL $TIMEOUT $*
	if [ $? -ne 0 ]; then
		echo "Command $* failed"
		cat $ERRORLOG
		cat $FUZZLOG
		exit 1;
	fi
	return $!
}

if [ ! -d $PBX_CACHE ];
then
	mkdir $PBX_CACHE
fi

# compilation
pushd $BUILDDIR
lcov --directory . --zerocounters
if [ ! -d $LOGDIR ];
then
	mkdir $LOGDIR
fi
if [ ! -d build ];
then
	mkdir build
fi
pushd build
cmake -DTENNCOR_TEST=ON ..
cmake --build .
popd

BINDIR=$BUILDDIR/bin/bin

ls $LOGDIR
# valgrind check
for _ in {1..5}
do
	echo "running tenncortest with memcheck"
	assert_cmd "valgrind --tool=memcheck $BINDIR/tenncortest --gtest_break_on_failure --gtest_shuffle > $ERRORLOG"
	echo "tenncortest valgrind check complete"
done

# tenncor tests
for _ in {1..9}
do
	echo "running batch of 3 tenncortest"
	assert_cmd "$BINDIR/tenncortest --gtest_break_on_failure --gtest_repeat=3 --gtest_shuffle > ${ERRORLOG}"
	echo "still running! 3 tests complete"
done

rm $ERRORLOG

# rocnnet demos
echo "running gd_demo"
assert_cmd "valgrind --tool=memcheck $BINDIR/gd_demo -o $PBX_CACHE -r 5 -t 1"
assert_cmd "$BINDIR/gd_demo -o $PBX_CACHE"
echo "gd_demo complete"

echo "running C++ dq_demo"
assert_cmd "valgrind --tool=memcheck $BINDIR/dq_demo -o $PBX_CACHE -e 1 -m 5"
assert_cmd "$BINDIR/dq_demo -o $PBX_CACHE"
echo "C++ dq_demo complete"

echo "running python dq_demo"
assert_cmd "python $BUILDDIR/pydemo/dq_demo.py"
echo "python dq_demo complete"

# coverage analysis
pushd $BUILDDIR
lcov --version
gcov --version
lcov --directory . --gcov-tool gcov-6 --capture --output-file coverage.info # capture coverage info
lcov --remove coverage.info '**/gtest*' '**/tests/*' '**/ipython/*' '/usr/*' --output-file coverage.info # filter out system and test code
lcov --list coverage.info # debug < see coverage here
on_travis coveralls-lcov --repo-token $COVERALLS_TOKEN coverage.info # uploads to coveralls
