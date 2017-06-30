//
// Created by Mingkai Chen on 2017-04-27.
//

#ifdef TENNCOR_FUTILS_HPP

namespace nnet
{

template <typename T>
varptr<T> sigmoid (varptr<T> x)
{
	return (T)1 / ((T)1 + exp(-x));
}

template <typename T>
varptr<T> tanh (varptr<T> x)
{
	varptr<T> etx = exp((T)2 * x);
	return (etx + (T)1) / (etx - (T)1);
}

}

#endif