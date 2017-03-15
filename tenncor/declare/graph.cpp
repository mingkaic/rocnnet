//
// Created by Mingkai Chen on 2017-03-12.
//

#include "graph/inode.hpp"
#include "graph/leaf/ileaf.hpp"
#include "graph/leaf/ivariable.hpp"
#include "graph/leaf/constant.hpp"
#include "graph/leaf/variable.hpp"
#include "graph/leaf/placeholder.hpp"
#include "graph/operation/iconnector.hpp"
#include "graph/operation/immutable/immutable.hpp"
#include "graph/operation/immutable/operation.hpp"
#include "graph/operation/immutable/matmul.hpp"
#include "graph/operation/immutable/elementary.hpp"
#include "graph/operation/immutable/transform.hpp"
#include "graph/varptr.hpp"

namespace nnet
{

// Instantiate tensors for instrumentation
template class varptr<double>;

template class placeptr<double>;

template class inode<double>;

template class ileaf<double>;

template class ivariable<double>;

template class constant<double>;

template class variable<double>;

template class placeholder<double>;

template class iconnector<double>;

template class immutable<double>;

template class operation<double>;

template class matmul<double>;

template varptr<double> operator + (const varptr<double> a);

template varptr<double> operator - (const varptr<double> a);

template varptr<double> sin (const varptr<double> a);

template varptr<double> cos (const varptr<double> a);

template varptr<double> tan (const varptr<double> a);

template varptr<double> csc (const varptr<double> a);

template varptr<double> sec (const varptr<double> a);

template varptr<double> cot (const varptr<double> a);

template varptr<double> exp (const varptr<double> a);

//template varptr<double> sqrt (const varptr<double> a); // TODO implement
//
//template varptr<double> pow (const varptr<double> a, double scalar); // TODO implement

template varptr<double> clip_val (const varptr<double> a, double min, double max);

template varptr<double> clip_norm (const varptr<double> a, double cap);

template varptr<double> operator + (double a, const varptr<double> b);

template varptr<double> operator + (const varptr<double> a, double b);

template varptr<double> operator + (const varptr<double> a, const varptr<double> b);

template varptr<double> operator - (double a, const varptr<double> b);

template varptr<double> operator - (const varptr<double> a, double b);

template varptr<double> operator - (const varptr<double> a, const varptr<double> b);

template varptr<double> operator * (double a, const varptr<double> b);

template varptr<double> operator * (const varptr<double> a, double b);

template varptr<double> operator * (const varptr<double> a, const varptr<double> b);

template varptr<double> operator / (double a, const varptr<double> b);

template varptr<double> operator / (const varptr<double> a, double b);

template varptr<double> operator / (const varptr<double> a, const varptr<double> b);

template varptr<double> transpose (const varptr<double> a);

template varptr<double> fit (const varptr<double> a, const varptr<double> watch);

template varptr<double> extend (const varptr<double> a, size_t index, size_t multiplier);

template varptr<double> compress (const varptr<double> a, int index = -1,
	std::function<double(const std::vector<double>&)> collector = mean<double>);

}