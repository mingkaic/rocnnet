#!/usr/bin/env bash
#
# purpose:
# this script tests tenncor and rocnnet libraries
# through gtest and demos
#

# ===== Define Installation Variables =====

FUZZLOG=fuzz.out
TIMEOUT=720 # 12 minute limit

BASEDIR=$1
BINDIR=$BASEDIR/bin/bin

# ===== Define Functions =====

assert_cmd() {
	eval timeout -s SIGKILL $TIMEOUT $*
	if [ $? -ne 0 ]; then
		echo "Command $* failed"
		cat $FUZZLOG
		exit 1;
	fi
	return $!
}

lcov --base-directory . --directory . --zerocounters

# ===== Run Tenncor Gtest =====

# valgrind check
for _ in {1..5}
do
	echo "running tenncortest with memcheck"
	assert_cmd "valgrind --tool=memcheck $BINDIR/tenncortest --gtest_break_on_failure --gtest_shuffle"
	echo "tenncortest valgrind check complete"
done

# tenncor tests
for _ in {1..9}
do
	echo "running batch of 3 tenncortest"
	assert_cmd "$BINDIR/tenncortest --gtest_break_on_failure --gtest_repeat=3 --gtest_shuffle"
	echo "still running! 3 tests complete"
done

# ===== Run Rocnnet Demos =====

# rocnnet demos
echo "running gd_demo"
assert_cmd "valgrind --tool=memcheck $BINDIR/gd_demo -o $BASEDIR/prebuilt_models -r 5 -t 1"
assert_cmd "$BINDIR/gd_demo -o $BASEDIR/prebuilt_models"
echo "gd_demo complete"

echo "running C++ dqn_demo"
assert_cmd "valgrind --tool=memcheck $BINDIR/dqn_demo -o $BASEDIR/prebuilt_models -e 1 -m 5"
assert_cmd "$BINDIR/dqn_demo -o $BASEDIR/prebuilt_models"
echo "C++ dqn_demo complete"

echo "running python dqn_demo"
assert_cmd "python $BASEDIR/pydemo/dqn_demo.py"
echo "python dqn_demo complete"

echo "running rbm_demo"
assert_cmd "valgrind --tool=memcheck $BINDIR/rbm_demo"
assert_cmd "$BINDIR/rbm_demo"
echo "rbm_demo complete"

#echo "running dbn_demo"
#assert_cmd "valgrind --tool=memcheck $BINDIR/dbn_demo -k 1 -n 30"
#assert_cmd "$BINDIR/dbn_demo -k 1"
#echo "dbn_demo complete"

# ===== Coverage Analysis ======
lcov --version
gcov --version
lcov --base-directory . --directory . --gcov-tool gcov-6 --capture --output-file coverage.info # capture coverage info
# filter out system and test code
lcov --remove coverage.info '**/gtest*' '**/tests/*' '**/ipython/*' '**/bin/*' '**/build/*' '/usr/include/*' --output-file coverage.info
lcov --list coverage.info # debug < see coverage here

if ! [ -z "$COVERALLS_TOKEN" ];
then
    echo "branched from $TRAVIS_BRANCH"
    coveralls-lcov --repo-token ${COVERALLS_TOKEN} coverage.info # uploads to coveralls
fi

echo "";
echo "============ ROCNNET TEST SUCCESS============";
