#!/usr/bin/env bash

DEMO=${1:-gd_demo}
./app/bin/bin/graph_gui & ./app/bin/bin/$DEMO
