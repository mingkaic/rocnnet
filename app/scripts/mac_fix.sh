#!/usr/bin/env bash

set -e

PYLIB_PATH=$1
DQN_SO=$2/bin/lib/_dqn.so

echo "linking to python2.7 in $ANACONDA_PATH"
install_name_tool -change libpython2.7.dylib $PYLIB_PATH $DQN_SO
otool -L $DQN_SO
