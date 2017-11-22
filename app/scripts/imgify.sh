#!/usr/bin/env bash
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python $THIS_DIR/csv_to_png.py $THIS_DIR/../bin/bin/op-profile.csv > op-profile.svg
