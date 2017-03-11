/*!
 *
 *  elementary.hpp
 *  cnnet
 *
 *  Purpose:
 *  elementary operators that wraps
 *  nodes in operation node
 *  using element wise transfer functions
 *
 *  Created by Mingkai Chen on 2016-10-24.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/operation/immutable/operation.hpp"
#include "graph/leaf/constant.hpp"

#pragma once
#ifndef TENNCOR_ELEMENTARY_HPP
#define TENNCOR_ELEMENTARY_HPP

namespace nnet
{

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
varptr<T> sqrt (const varptr<T> a); // TODO implement

//! a to the power of scalar
template <typename T>
varptr<T> pow (const varptr<T> a, T scalar); // TODO implement

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

}

#include "../../../../src/graph/operation/immutable/elementary.ipp"

#endif /* TENNCOR_ELEMENTARY_HPP */
