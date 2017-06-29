/*!
 *
 *  futils.hpp
 *  cnnet
 *
 *  Purpose:
 *  define commonly used activation functions
 *  and useful graph operations
 *
 *  Created by Mingkai Chen on 2016-09-30.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/connector/immutable/elementary.hpp"
#include "graph/varptr.hpp"

#ifndef TENNCOR_FUTILS_HPP
#define TENNCOR_FUTILS_HPP

namespace nnet
{

//! sigmoid function: f(x) = 1/(1+e^-x)
template <typename T>
varptr<T> sigmoid (varptr<T> x);

//! tanh function: f(x) = (e^(2*x)+1)/(e^(2*x)-1)
template <typename T>
varptr<T> tanh (varptr<T> x);

}

#include "../../src/utils/futils.ipp"

#endif /* TENNCOR_FUTILS_HPP */
