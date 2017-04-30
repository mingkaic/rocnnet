/*!
 *
 *  transform.hpp
 *  cnnet
 *
 *  Purpose:
 *  transform operators that wraps
 *  nodes in operation node
 *  that reshapes arguments
 *
 *  Created by Mingkai Chen on 2016-10-24.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/connector/immutable/immutable.hpp"

#ifndef TENNCOR_TRANSFORM_HPP
#define TENNCOR_TRANSFORM_HPP

namespace nnet
{

//! transpose a
template <typename T>
varptr<T> transpose (const varptr<T> a);

//! fit data in a to watch's shape, ignores all jacobian (todo: change to selectively ignore watch's jacobian)
//! watch needs to be a dependency of the resulting node,
//! because shape changes to watch should trigger shape update for output node
template <typename T>
varptr<T> fit (const varptr<T> a, const varptr<T> watch);

//! extend data in a to along index dimension multiplier times
template <typename T>
varptr<T> extend (const varptr<T> a, size_t index, size_t multiplier);

//! compresses data along dimensions specified by index
//! -1 index compresses all elements in the tensor (output is a scalar)
template <typename T>
varptr<T> compress (const varptr<T> a, int index,
	std::function<T(const std::vector<T>&)> collector);

// Dimensionality Reduction Functions (Wrappers for compress)
//! compress tensor by taking maximum value across specified dimension
//! -1 dimension obtains maximum value in the entire tensor
template <typename T>
varptr<T> reduce_max (const varptr<T> a, int dimension = -1);

//! compress tensor by taking the sum of values across specified dimension(s)
//! -1 dimension obtains the sum of all values in the entire tensor
template <typename T>
varptr<T> reduce_sum (const varptr<T> a, int dimension = -1);

//! compress tensor by taking the mean of values across specified dimension(s)
//! -1 dimension obtains the mean of values in the entire tensor
template <typename T>
varptr<T> reduce_mean (const varptr<T> a, int dimension = -1);

//! compresses data along dimensions specified by dimension
//! by taking the index using the compare function
//! -1 index compresses all elements in the tensor (output is a scalar)
//! takes left argument of compare if compare evaluates to true
template <typename T>
varptr<T> arg_compress (const varptr<T> a, int dimension,
	std::function<bool(T,T)> compare);

//! obtains the indices of the maximum value across specified dimension
//! -1 index looks returns a vector coordinate specifying max value in tensor a
template <typename T>
varptr<T> arg_max (const varptr<T> a, int dimension = -1);

//! trace of a
// todo: implement [grad(trace(f(x)), x) = transpose(scalar_grad(f(x), x))]
template <typename T>
varptr<T> trace (const varptr<T> a);

//! inverse of matrix a
// todo: implement
template <typename T>
varptr<T> inverse (const varptr<T> a);

}

#include "../../../../src/graph/connector/immutable/transform.ipp"

#endif /* TENNCOR_TRANSFORM_HPP */
