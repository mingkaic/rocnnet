[![Build Status](https://travis-ci.org/mingkaic/rocnnet.svg?branch=master)](https://travis-ci.org/mingkaic/rocnnet)
[![Coverage Status](https://coveralls.io/repos/github/mingkaic/rocnnet/badge.svg?branch=master)](https://coveralls.io/github/mingkaic/rocnnet?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/mingkaic/rocnnet/badge.svg?branch=HEAD)](https://coveralls.io/github/mingkaic/rocnnet?branch=HEAD)

## Synopsis

ROCNNet is a neural net library implemented in C++ using [Tenncor](https://github.com/mingkaic/rocnnet/blob/master/app/tenncor/README.md) for automatic differentiation.

## Build

CMake 3.6 is required.

Download cmake: https://cmake.org/download/

Building from another directory is recommended:

	mkdir build 
	cd build
	cmake <flags> <path/to/rocnnet>
	cmake --build .

Binaries and libraries should be found in the /bin directory

Flags include the following:

- TENNCOR_TEST=<ON/OFF> (build tests)
- LCOVERAGE=(ON/OFF) (build with coverage)
- CSV_RCD=(ON/OFF) (enable graph structure recording to csv file)
- RPC_RCD=(ON/OFF) (enable graph realtime inspection via grpc)
- SWIG_DQN=<ON/OFF> (build python wrapper for dqn_agent)

## Visualization

During executation, call `static_cast<rocnnet_record::csv_record*>(rocnnet_record::record_status::rec.get())->to_csv<double>()` to print all nodes out to a csv file, `op-profile.csv`,
recording all nodes and connections.

To visualize the graph, install [graphviz (and its dependencies)](https://pygraphviz.github.io/documentation/pygraphviz-1.3rc1/install.html),
then run 

	bash /path/to/app/scripts/imgify.sh
	
To run realtime graph inspection gui, install qt5 and update submodule. Included as a utility, run

	bash /path/to/app/scripts/qt_setup.sh

## Demos

DQN Demos taken from https://github.com/siemanko/tensorflow-deepq

To demonstrate correctness, I have setup two examples:

- multilayer perceptron (mlp) using vanilla gradient descent learner learning to average second number.
it has the following graph:

![alt tag](https://github.com/mingkaic/rocnnet/blob/master/imgs/gd_graph.png)

- deep q-network (dqn) using rmsprop learner learning with the same test as mlp except it takes the maximum average and takes the distance 
from taken average and expected average as the error. reward is given inversely proportional to the error (lower the error, 
higher the reward with the error clipped between -1 and 1)
it has the following graph:

![alt tag](https://github.com/mingkaic/rocnnet/blob/master/imgs/dqn_graph.png)

- restricted boltzmann machine using K-Contrastive Divergence (CD) or K-Persistent CD (PCD) to speedup gibb sampling. 
Gibbb sampling in RBM forces hidden states towards something most representative of the observed probability distribution.
Since CD can recurse K times, the graph is too large to show.
Instead here is the reconstructed input on the popular [mnist dataset](http://www-labs.iro.umontreal.ca/~lisa/deep/data/mnist/):

![alt tag](https://github.com/mingkaic/rocnnet/blob/master/imgs/rbm_epoch-1.png)
after 1 epoch of sampling, k=15

![alt tag](https://github.com/mingkaic/rocnnet/blob/master/imgs/rbm_random.png)
reconstruction from random instantiation
