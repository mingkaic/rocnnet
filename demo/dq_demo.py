#!/usr/bin/env python

import _init_paths
from dqn import dqn_agent as agent

net = agent(10, [5, 5, 5], 0.9, 'pybrain')
