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

## License

MIT License

Copyright (c) 2016 mingkaic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
