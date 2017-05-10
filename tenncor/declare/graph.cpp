//
// Created by Mingkai Chen on 2017-03-12.
//

#include "graph/inode.hpp"
#include "graph/leaf/ileaf.hpp"
#include "graph/leaf/ivariable.hpp"
#include "graph/leaf/constant.hpp"
#include "graph/leaf/variable.hpp"
#include "graph/leaf/placeholder.hpp"
#include "graph/connector/iconnector.hpp"
#include "graph/connector/immutable/immutable.hpp"
#include "graph/connector/immutable/matmul.hpp"
#include "graph/connector/immutable/elementary.hpp"
#include "graph/connector/immutable/transform.hpp"
#include "graph/varptr.hpp"

namespace nnet
{

// Instantiate tensors for instrumentation
template class varptr<double>;

template class placeptr<double>;

template class constant<double>;

template class variable<double>;

template class placeholder<double>;

template class immutable<double>;

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

template varptr<double> compress (const varptr<double> a, optional<size_t> index,
	std::function<double(const std::vector<double>&)> collector);

template varptr<double> reduce_max (const varptr<double> a, optional<size_t> dimension = optional<size_t>());

template varptr<double> reduce_sum (const varptr<double> a, optional<size_t> dimension = optional<size_t>());

template varptr<double> reduce_mean (const varptr<double> a, optional<size_t> dimension = optional<size_t>());

template varptr<double> arg_compress (const varptr<double> a, optional<size_t> dimension,
	std::function<size_t(const std::vector<double>&)> compare);

template varptr<double> arg_max (const varptr<double> a, optional<size_t> dimension = optional<size_t>());

//template varptr<double> trace (const varptr<double> a);

//template varptr<double> inverse (const varptr<double> a);

}
