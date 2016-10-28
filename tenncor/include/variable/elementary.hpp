//
//  elementary.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-24.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "operation.hpp"

#ifndef elementary_hpp
#define elementary_hpp

namespace nnet {

// operators that will replace elementary operation objects
template<typename T>
VAR_PTR<T> operator + (const VAR_PTR<T>& a);

template<typename T>
VAR_PTR<T> operator - (const VAR_PTR<T>& a);

template<typename T>
VAR_PTR<T> sin (const VAR_PTR<T>& a);

template<typename T>
VAR_PTR<T> cos (const VAR_PTR<T>& a);

template<typename T>
VAR_PTR<T> tan (const VAR_PTR<T>& a);

template<typename T>
VAR_PTR<T> csc (const VAR_PTR<T>& a);

template<typename T>
VAR_PTR<T> sec (const VAR_PTR<T>& a);

template<typename T>
VAR_PTR<T> cot (const VAR_PTR<T>& a);

template<typename T>
VAR_PTR<T> exp (const VAR_PTR<T>& a);

template<typename T>
VAR_PTR<T> operator + (T a, const VAR_PTR<T>& b);

template<typename T>
VAR_PTR<T> operator + (const VAR_PTR<T>& a, T b);

template<typename T>
VAR_PTR<T> operator + (const VAR_PTR<T>& a, const VAR_PTR<T>& b);

template<typename T>
VAR_PTR<T> operator - (T a, const VAR_PTR<T>& b);

template<typename T>
VAR_PTR<T> operator - (const VAR_PTR<T>& a, T b);

template<typename T>
VAR_PTR<T> operator - (const VAR_PTR<T> &a, const VAR_PTR<T> &b);

template<typename T>
VAR_PTR<T> operator * (T a, const VAR_PTR<T>& b);

template<typename T>
VAR_PTR<T> operator * (const VAR_PTR<T>& a, T b);

template<typename T>
VAR_PTR<T> operator * (const VAR_PTR<T> &a, const VAR_PTR<T> &b);

template<typename T>
VAR_PTR<T> operator / (T a, const VAR_PTR<T>& b);

template<typename T>
VAR_PTR<T> operator / (const VAR_PTR<T>& a, T b);

template<typename T>
VAR_PTR<T> operator / (const VAR_PTR<T> &a, const VAR_PTR<T> &b);

}

#include "../../src/variable/elementary.tpp"

#endif /* elementary_hpp */
