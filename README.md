## Synopsis

ROCNNet is a general purpose tensor-based automatic differentiation calculation, and machine learning library implemented in C++.
Tensors are wrapped in variable nodes which form parts of reusable graph operations. Information propogate through this graph reactively, 
meaning updates of individual nodes will trigger updates to corresponding operations when necessary.

### Library

There are two libraries:

- Tenncor, the tensor calculation library

- Rocnnet, simple perceptron, and neural nets library

## Examples

#### Tenncor Library

    #include "executor/varptr.hpp"
    #include "executor/gradient.hpp"
    #include "graph/variable/variable.hpp"
    #include "graph/operation/special/matmul.hpp"
    #include "graph/operation/function.hpp"
    
    using namespace nnet;
    
    int main () {
        tensorshape common = std::vector<size_t>{5, 5};
        random_uniform<double> rinit(-1, 1);
	    session& sess = session::get_instance();
    
        // initializes a 5 by 5 matrix with uniformly distributed
        // doubles between -1 and 1
        varptr<double> A = new variable<double>(common, rinit, "a");
        placeptr<double> B = new placeholder<double>(common, "b");
        varptr<double> C = matmul<double>::build(A, B);
        varptr<double> D = sigmoid<double>(C);
        
        sess.initialize_all<double>();
        B = std::vector<double>{...};
        
        gradient<double>* grad(D);
        // prevent changes to B from cascading to gradient value
        grad.freeze();
        grad.execute();
        
        // forward accumulation
        tensor<double>* result = D->get_eval();
        // reverse accumulation
        tensor<double>* grad_result;
        grad.collect_grad(
        [&grad_result](ivariable<double>* key, 
                       placeholder<double>* value)
        {
            grad_result = value->get_eval();
        });
        
        delete A;
        delete B;
    } 

## Build

CMake 3.6 is required.

Download cmake: https://cmake.org/download/

Building from another directory is recommended:

    mkdir build 
    cd build
    cmake ..
    make

Binaries and libraries should be found in the /bin directory

## API Reference

Working in Progress (Using doxygen)

## Components

#### Tenncor

Tenncor library holds generic classes for tensor calculation.

At the lowest layer, tensors comprise of two components: raw data and tensor shape. 
Raw data are allocated by Allocators objects supplied as a template parameter to the Tensor.

A tensor is simply an N-dimensional container. 
A matrix is a 2-d container since it could be visualized as arrays of vectors which are 1-d containers.
To clarify some terminology, the rank of a tensor is the maximum tensor's dimensionality (the N in N-dimension).
The index is some dimensionality that is lower or equal to the rank; that is, for a 3 rank tensor, indices 0, 1, and 2 are possible.
Dimensional value at index i is however many tensors of rank i-1 can potentially fit into the "column" at dimension i.
For instance, the number of rows in a matrix can the dimensional value at index 0 or 1 (mathematical convention says 0, tenncor convention says 1).
Also for matrices, rows are considered the y-coordinate, and columns the x-coordinate, so rows inhabit dimension-2 (index 1), while columns are dimension-1 (index 0).

Shapes can be in any of 3 states: fully defined, partially defined, and undefined. By default, a tensor is undefined.
Undefined tensors can potentially perform operations with any other tensors regardless of dimensional value or rank.
Undefined tensors are not always desirable since it is often wise to verify shape compatibility before making an expensive calculation.
Partially defined shapes has a definite rank, but potentially unspecified dimensional values.
Fully defined shapes have specified rank and dimensional values.
An unspecified dimensional value is simply 0.

Tensors are wrapped in variable nodes hold points to gradient nodes and corresponding operations.
Variable and operation nodes are partitioned by composite pattern; operations and leaf nodes inherit from a variable interface.

#### Rocnnet

Some implemented mechanisms include:
* multilayer perceptron using linear regression
* deep q-neural nets (incomplete) https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

## Contributors

Ming Kai Chen