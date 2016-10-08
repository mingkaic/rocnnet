//
//  shared_ops.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-09-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef shared_ops_hpp
#define shared_ops_hpp

namespace shared_cnnet {

template <typename T>
static inline T identity (T in) { return in; }

template <typename T>
static inline T op_neg (T in) { return -in; }

// achieve numerical stability for trig functions
template <typename T>
static inline T op_sin (T in) { return std::sin(in); }

template <typename T>
static inline T op_cos (T in) { return std::cos(in); }

template <typename T>
static inline T op_tan (T in) { return std::tan(in); }

template <typename T>
static inline T op_csc (T in) { return 1/std::sin(in); }

template <typename T>
static inline T op_sec (T in) { return 1/std::cos(in); }

template <typename T>
static inline T op_cot (T in) { return std::cos(in)/std::sin(in); }

template <typename T>
static inline T op_exp (T in) { return std::exp(in); }

template <typename T>
static inline T op_add (T a, T b) { return a + b; }

template <typename T>
static inline T op_sub (T a, T b) { return a - b; }

template <typename T>
static inline T op_mul (T a, T b) { return a * b; }

template <typename T>
static inline T op_div (T a, T b) { return a / b; }

}

#endif /* shared_ops_hpp */
