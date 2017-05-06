#!/usr/bin/env python

import _init_paths
from dqn import *

hiddens = [5, 5, 5]
net = dqn_agent(10, hiddens, 0.9, 'pybrain')
