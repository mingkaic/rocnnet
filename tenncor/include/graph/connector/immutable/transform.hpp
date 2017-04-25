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

//! obtain the mean of input data
template <typename T>
T mean (const std::vector<T>& data);

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

//! compression of index -1 compresses all elements in a (result is a scalar)
template <typename T>
varptr<T> compress (const varptr<T> a, int index = -1,
	std::function<T(const std::vector<T>&)> collector = mean<T>);

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
