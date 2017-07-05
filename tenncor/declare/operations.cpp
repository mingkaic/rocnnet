//
// Created by Mingkai Chen on 2017-06-28.
//

#include "graph/operations/operations.hpp"

namespace nnet
{

template varptr<double> operator + (const varptr<double> a);

template varptr<double> operator - (const varptr<double> a);

template varptr<double> sin (const varptr<double> a);

template varptr<double> cos (const varptr<double> a);

template varptr<double> tan (const varptr<double> a);

template varptr<double> csc (const varptr<double> a);

template varptr<double> sec (const varptr<double> a);

template varptr<double> cot (const varptr<double> a);

template varptr<double> exp (const varptr<double> a);

template varptr<double> sqrt (const varptr<double> a);

template varptr<double> pow (const varptr<double> a, double scalar);

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

template varptr<double> transpose (const varptr<double> a, std::pair<size_t,size_t> axis_swap = {0, 1});

template varptr<double> fit (const varptr<double> a, const varptr<double> watch);

template varptr<double> extend (const varptr<double> a, size_t index, size_t multiplier);

template varptr<double> compress (const varptr<double> a, optional<size_t> index,
	ELEM_FUNC<double> collector, std::string name = "compress");

template varptr<double> reduce_max (const varptr<double> a, optional<size_t> dimension = optional<size_t>());

template varptr<double> reduce_sum (const varptr<double> a, optional<size_t> dimension = optional<size_t>());

template varptr<double> reduce_mean (const varptr<double> a, optional<size_t> dimension = optional<size_t>());

template varptr<double> arg_compress (const varptr<double> a, optional<size_t> dimension,
	ELEM_FUNC<double> compare, std::string name = "argcompress");

template varptr<double> arg_max (const varptr<double> a, optional<size_t> dimension = optional<size_t>());

//template varptr<double> trace (const varptr<double> a);

//template varptr<double> inverse (const varptr<double> a);

}