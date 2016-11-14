## Synopsis

CNNet is a general purpose tensor manipulation, and machine learning library implemented in C++11. (not a library yet)
Aside from a compiler supporting C++11 requirement, no dependencies are required.

### Components

There are two components to this library:
- tensor manipulation
	* Like tensorflow, main actors are variables, tensors, and operations.
	* Variables hold tensors.
	* Operations operate on variables. Operations are differentiable.

- machine learning
Some implemented mechanisms include:
	* toy multilayer perceptron using linear regression
	* deep q-neural nets https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

## Examples

None

## Motivation

Tensor Component:
Tensorflow's C++ API as of late is hilariously underwhelming for obvious reasons.
(C++ code on tensors is very ugly.)
This is my attempt at a tensor library that is less ugly (to me at least).
Will consider boost to beautify and enforce stability in the future.

Neural Net Component:
Use tensor component to implement various neural nets and machine learning mechanisms
for demonstration and use on higher level machine learning systems.

## Installation

Not a library yet

## API Reference

Working in Progress (Using doxygen)

## Tests

### For Mac and Linux:

`~$ make test` builds and runs an application that tests all features in the project

`~$ make test_ten` builds and runs a test for tensorflow clone features

`~$ make test_net` builds and runs a test for neural net features

### For Windows:

Nothing yet ;)

## Contributors

No issue tracker, external websites yet