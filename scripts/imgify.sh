#!/usr/bin/env bash
python csv_to_png.py ../bin/bin/op-profile.csv > op-profile.svg
dot -Tpng op-profile.svg > op-graph.png
