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

namespace nnet
{

#pragma once
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

//! a to the power of scalar
template <typename T>
varptr<T> pow (const varptr<T> a, double scalar);

//! clip values in range [min, max]
template<typename T>
varptr<T> clip_val (const varptr<T> a, T min, T max);

//! normalize clip values with capacity cap
template <typename T>
varptr<T> clip_norm (const varptr<T> a, T cap);

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

#pragma once
#ifndef TENNCOR_TRANSFORM_HPP
#define TENNCOR_TRANSFORM_HPP

//! transpose a along first 2 dimension
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
//! -1 index compresses all elements in the tensor (output is a scalar)
template <typename T>
varptr<T> compress (const varptr<T> a, optional<size_t> index,
	ELEM_FUNC<T> collector, std::string name = "compress",
	std::function<varptr<T>(varptr<T>,varptr<T>)> bprop =
	[](varptr<T> back, varptr<T>){ return back; });

// Dimensionality Reduction Functions (Wrappers for compress)
//! compress tensor by taking maximum value across specified dimension
//! -1 dimension obtains maximum value in the entire tensor
template <typename T>
varptr<T> reduce_max (const varptr<T> a, optional<size_t> dimension = optional<size_t>());

//! compress tensor by taking the sum of values across specified dimension(s)
//! -1 dimension obtains the sum of all values in the entire tensor
template <typename T>
varptr<T> reduce_sum (const varptr<T> a, optional<size_t> dimension = optional<size_t>());

//! compress tensor by taking the mean of values across specified dimension(s)
//! -1 dimension obtains the mean of values in the entire tensor
template <typename T>
varptr<T> reduce_mean (const varptr<T> a, optional<size_t> dimension = optional<size_t>());

//! compresses data along dimensions specified by dimension
//! by taking the index using the compare function
//! -1 index compresses all elements in the tensor (output is a scalar)
//! takes left argument of compare if compare evaluates to true
template <typename T>
varptr<T> arg_compress (const varptr<T> a, optional<size_t> dimension,
	ELEM_FUNC<T> compare, std::string name = "argcompress");

//! obtains the indices of the maximum value across specified dimension
//! -1 index looks returns a vector coordinate specifying max value in tensor a
template <typename T>
varptr<T> arg_max (const varptr<T> a, optional<size_t> dimension = optional<size_t>());

// todo: implement
// [grad(trace(f(x)), x) = transpose(scalar_grad(f(x), x))]
//! trace of a
template <typename T>
varptr<T> trace (const varptr<T> a);

//! inverse of matrix a
template <typename T>
varptr<T> inverse (const varptr<T> a);

#endif /* TENNCOR_TRANSFORM_HPP */

}

#include "../../../src/graph/operations/elementary.ipp"

#include "../../../src/graph/operations/transform.ipp"

