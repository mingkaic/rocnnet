/*!
 *
 *  operations.hpp
 *  cnnet
 *
 *  Purpose:
 *  elementary operators that wraps
 *  nodes in operation node
 *  using element wise transfer functions
 *
 *  transform operators that wraps
 *  nodes in operation node
 *  that reshapes arguments
 *
 *  Created by Mingkai Chen on 2016-10-24.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/connector/immutable/immutable.hpp"
#include "graph/leaf/constant.hpp"
#include "graph/operations/operation_utils.hpp"

#pragma once

template <typename T>
using AGGREGATE = std::function<T(T, T)>;

template <typename T>
using REDUCE = std::function<T(std::vector<T>)>;

namespace nnet
{

#ifndef TENNCOR_ELEMENTARY_HPP
#define TENNCOR_ELEMENTARY_HPP

//! wraps an empty node, usually to avoid overlapping references
template <typename T>
varptr<T> identity (varptr<T> x);

//! absolute value of a
template<typename T>
varptr<T> operator + (const varptr<T> a);

//! negative value of a
template<typename T>
varptr<T> operator - (const varptr<T> a);

//! sin of a
template<typename T>
varptr<T> sin (const varptr<T> a);

//! cos of a
template<typename T>
varptr<T> cos (const varptr<T> a);

//! tan of a
template<typename T>
varptr<T> tan (const varptr<T> a);

//! cosecant of a
template<typename T>
varptr<T> csc (const varptr<T> a);

//! secant of a
template<typename T>
varptr<T> sec (const varptr<T> a);

//! cotangent of a
template<typename T>
varptr<T> cot (const varptr<T> a);

//! e of power a
template<typename T>
varptr<T> exp (const varptr<T> a);

//! square root of a
template <typename T>
varptr<T> sqrt (const varptr<T> a);

//! round a
template <typename T>
varptr<T> round (const varptr<T> a);

//! natural log a
template <typename T>
varptr<T> log (const varptr<T> a);

//! a to the power of scalar
template <typename T>
varptr<T> pow (const varptr<T> a, double scalar);

//! clip values in range [min, max]
template<typename T>
varptr<T> clip_val (const varptr<T> a, T min, T max);

//! normalize clip values with capacity cap
template <typename T>
varptr<T> clip_norm (const varptr<T> a, T cap);

//! output value 0 if false == compare(a, b) else 1 for each element
template <typename T>
varptr<T> conditional (T a, const varptr<T> b, std::function<bool(T,T)> compare, std::string name);

template <typename T>
varptr<T> conditional (const varptr<T> a, T b, std::function<bool(T,T)> compare, std::string name);

template <typename T>
varptr<T> conditional (const varptr<T> a, const varptr<T> b, std::function<bool(T,T)> compare, std::string name);

//! sample using binominal distribution given tensors (or scalars) n and p
// todo: after implementing type, restrict n to integers
template <typename T>
varptr<T> binomial_sample (T n, const varptr<T> p);

template <typename T>
varptr<T> binomial_sample (const varptr<T> n, T p);

template <typename T>
varptr<T> binomial_sample (const varptr<T> n, const varptr<T> p);

//! add scalar a and b
template<typename T>
varptr<T> operator + (T a, const varptr<T> b);

//! add a and scalar b
template<typename T>
varptr<T> operator + (const varptr<T> a, T b);

//! add a and b
template<typename T>
varptr<T> operator + (const varptr<T> a, const varptr<T> b);

//! subtract scalar a and b
template<typename T>
varptr<T> operator - (T a, const varptr<T> b);

//! subtract a and scalar b
template<typename T>
varptr<T> operator - (const varptr<T> a, T b);

//! subtract a and b
template<typename T>
varptr<T> operator - (const varptr<T> a, const varptr<T> b);

//! multiply scalar a and b
template<typename T>
varptr<T> operator * (T a, const varptr<T> b);

//! multiply a and scalar b
template<typename T>
varptr<T> operator * (const varptr<T> a, T b);

//! multiply a and b
template<typename T>
varptr<T> operator * (const varptr<T> a, const varptr<T> b);

//! divide scalar a and b
template<typename T>
varptr<T> operator / (T a, const varptr<T> b);

//! divide a and scalar b
template<typename T>
varptr<T> operator / (const varptr<T> a, T b);

//! divide a and b
template<typename T>
varptr<T> operator / (const varptr<T> a, const varptr<T> b);

//! add a and b along a specific axis, dimension values outside of axis must match
template <typename T>
varptr<T> add_axial_a (const varptr<T> a, const varptr<T> b, size_t axis_a);

template <typename T>
varptr<T> add_axial_b (const varptr<T> a, const varptr<T> b, size_t axis_b);

//! subtract a and b along a specific axis, dimension values outside of axis must match
template <typename T>
varptr<T> sub_axial_a (const varptr<T> a, const varptr<T> b, size_t axis_a);

template <typename T>
varptr<T> sub_axial_b (const varptr<T> a, const varptr<T> b, size_t axis_b);

//! multiply a and b along a specific axis, dimension values outside of axis must match
template <typename T>
varptr<T> mul_axial_a (const varptr<T> a, const varptr<T> b, size_t axis_a);

template <typename T>
varptr<T> mul_axial_b (const varptr<T> a, const varptr<T> b, size_t axis_b);

//! divide a and b along a specific axis, dimension values outside of axis must match
template <typename T>
varptr<T> div_axial_a (const varptr<T> a, const varptr<T> b, size_t axis_a);

template <typename T>
varptr<T> div_axial_b (const varptr<T> a, const varptr<T> b, size_t axis_b);

#endif /* TENNCOR_ELEMENTARY_HPP */

#ifndef TENNCOR_TRANSFORM_HPP
#define TENNCOR_TRANSFORM_HPP

//! transpose a along first 2 dimension
// todo: check if axis_swap are the same dimensions, if so, return a as is (invalid transpose) + leave warning
template <typename T>
varptr<T> transpose (const varptr<T> a, std::pair<size_t,size_t> axis_swap = {0, 1});

//! fit data in a to watch's shape, ignores all jacobian (todo: change to selectively ignore watch's jacobian)
//! watch needs to be a dependency of the resulting node,
//! because shape changes to watch should trigger shape update for output node
template <typename T>
varptr<T> fit (const varptr<T> a, const varptr<T> watch);

//! extend data in a to along index dimension multiplier times
template <typename T>
varptr<T> extend (const varptr<T> a, size_t index, size_t multiplier);

//! compresses data along dimensions specified by index
//! unspecified index compresses all elements in the tensor (output is a scalar)
template <typename T>
varptr<T> compress (const varptr<T> a, AGGREGATE<T> collector,
	optional<size_t> index, std::string name = "compress");

// Dimensionality Reduction Functions (Wrappers for compress)
//! compress tensor by taking maximum value across specified dimension
//! unspecified dimension obtains maximum value in the entire tensor
template <typename T>
varptr<T> reduce_max (const varptr<T> a, optional<size_t> dimension = optional<size_t>());

//! compress tensor by taking the sum of values across specified dimension(s)
//! unspecified dimension obtains the sum of all values in the entire tensor
template <typename T>
varptr<T> reduce_sum (const varptr<T> a, optional<size_t> dimension = optional<size_t>());

//! compress tensor by taking the mean of values across specified dimension(s)
//! unspecified dimension obtains the mean of values in the entire tensor
template <typename T>
varptr<T> reduce_mean (const varptr<T> a, optional<size_t> dimension = optional<size_t>());

//! compresses data along dimensions specified by dimension
//! by taking the index using the compare function
//! unspecified dimension compresses all elements in the tensor (output is a scalar)
//! takes left argument of compare if compare evaluates to true
template <typename T>
varptr<T> arg_compress (const varptr<T> a, REDUCE<T> search,
	optional<size_t> dimension, std::string name = "argcompress");

//! obtains the indices of the maximum value across specified dimension
//! -1 index looks returns a vector coordinate specifying max value in tensor a
template <typename T>
varptr<T> arg_max (const varptr<T> a, optional<size_t> dimension = optional<size_t>());

//! flip a in specified dimensions
template <typename T>
varptr<T> flip (const varptr<T> a, std::vector<size_t> dims);

//! for example: window {0, 1} gives output f[i, j, :] = sum(a[i:i+filtshape[0], j:j+filtshape[1], :] * filter)
//! whereas window {0,2} gives output f[i, :, j] = sum(a[i:i+filtshape[0], :, j:j+filtshape[1]] * filter)
//! if pad == true, then pad output with zero to fit a's shape, otherwise leave as is after cross_corr
template <typename T>
varptr<T> cross_corr2d (const varptr<T> a, const varptr<T> filter, std::pair<size_t,size_t> dims = {0, 1});
	
//! convolve a with filter, conv(a, filter, dims) = cross_conv(a, flip(filter), dims)
template <typename T>
varptr<T> conv2d (const varptr<T> a, const varptr<T> filter, std::pair<size_t,size_t> dims = {0, 1});

// todo: implement
// [grad(trace(f(x)), x) = transpose(scalar_grad(f(x), x))]
//! trace of a
template <typename T>
varptr<T> trace (const varptr<T> a);

//! inverse of matrix a
template <typename T>
varptr<T> inverse (const varptr<T> a);

#endif /* TENNCOR_TRANSFORM_HPP */

#ifndef TENNCOR_MATMUL_HPP
#define TENNCOR_MATMUL_HPP

//! matrix multiplication (todo: expand to include matmul along other dimensions, currently {0, 1} only)
template <typename T>
varptr<T> matmul (const varptr<T> a, const varptr<T> b,
	bool transposeA = false, bool transposeB = false);

#endif /* TENNCOR_MATMUL_HPP */

}

#include "../../../src/graph/operations/elementary.ipp"

#include "../../../src/graph/operations/transform.ipp"

#include "../../../src/graph/operations/matmul.ipp"

