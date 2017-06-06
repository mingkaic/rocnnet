//
//  tensor.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2017-02-05.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_TENSOR_HANDLER_HPP

#include "memory/shared_rand.hpp"

namespace nnet
{

template <typename T>
itensor_handler<T>* itensor_handler<T>::clone (void) const
{
	return clone_impl();
}

template <typename T>
itensor_handler<T>* itensor_handler<T>::move (void)
{
	return move_impl();
}

template <typename T>
void itensor_handler<T>::operator () (tensor<T>*& out,
	std::vector<const tensor<T>*> args) const
{
	std::vector<tensorshape> ts;
	std::vector<const T*> raws;
	for (const tensor<T>* arg : args)
	{
		assert(arg && arg->is_alloc());
		ts.push_back(arg->get_shape());
		raws.push_back(arg->raw_data_);
	}
	tensorshape s = calc_shape(ts);
	if (nullptr == out)
	{
		out = new tensor<T>(s);
	}

	if (s.is_fully_defined())
	{
		// if out is allocated, verify shape with out
		if (out->is_alloc())
		{
			tensorshape oshape = out->get_shape();
			if (false == s.is_compatible_with(oshape))
			{
				std::stringstream ss;
				print_shape(s, ss);
				ss << " is incompatible with output shape ";
				print_shape(oshape, ss);
				throw std::runtime_error(ss.str());
			}
		}
		// otherwise allocate out
		else
		{
			out->allocate(s);
		}
		calc_data(out->raw_data_, s, raws, ts);
	}
}

template <typename T>
transfer_func<T>::transfer_func (SHAPER shaper, FORWARD_OP<T> forward) :
	shaper_(shaper), forward_(forward) {}

template <typename T>
tensorshape transfer_func<T>::calc_shape (std::vector<tensorshape> shapes) const
{
	return shaper_(shapes);
}

template <typename T>
transfer_func<T>* transfer_func<T>::clone (void) const
{
	return static_cast<transfer_func<T>*>(clone_impl());
}

template <typename T>
transfer_func<T>* transfer_func<T>::move (void)
{
	return static_cast<transfer_func<T>*>(move_impl());
}

template <typename T>
void transfer_func<T>::operator () (tensor<T>*& out, std::vector<const tensor<T>*> args) const
{
	itensor_handler<T>::operator () (out, args);
}

template <typename T>
itensor_handler<T>* transfer_func<T>::clone_impl (void) const
{
	return new transfer_func(*this);
}

template <typename T>
itensor_handler<T>* transfer_func<T>::move_impl (void)
{
	return new transfer_func(std::move(*this));
}

template <typename T>
void transfer_func<T>::calc_data (T* dest, const tensorshape& outshape,
	std::vector<const T*>& srcs, std::vector<tensorshape>& inshapes) const
{
	forward_(dest, outshape, srcs, inshapes);
}

template <typename T>
initializer<T>* initializer<T>::clone (void) const
{
	return static_cast<initializer<T>*>(this->clone_impl());
}

template <typename T>
initializer<T>* initializer<T>::move (void)
{
	return static_cast<initializer<T>*>(this->move_impl());
}

template <typename T>
void initializer<T>::operator () (tensor<T>*& out) const
{
	itensor_handler<T>::operator ()(out, {out});
}

template <typename T>
tensorshape initializer<T>::calc_shape (std::vector<tensorshape> shapes) const
{
	if (shapes.empty()) return {};
	return shapes[0];
}

template <typename T>
const_init<T>::const_init (T value) : value_(value) {}

template <typename T>
const_init<T>* const_init<T>::clone (void) const
{
	return static_cast<const_init<T>*>(clone_impl());
}

template <typename T>
const_init<T>* const_init<T>::move (void)
{
	return static_cast<const_init<T>*>(move_impl());
}

template <typename T>
itensor_handler<T>* const_init<T>::clone_impl (void) const
{
	return new const_init(*this);
}

template <typename T>
itensor_handler<T>* const_init<T>::move_impl (void)
{
	return new const_init(std::move(*this));
}

template <typename T>
void const_init<T>::calc_data (T* dest, const tensorshape& outshape,
	std::vector<const T*>&, std::vector<tensorshape>&) const
{
	size_t len = outshape.n_elems();
	std::fill(dest, dest+len, value_);
}

template <typename T>
rand_uniform<T>::rand_uniform (T min, T max) :
	distribution_(min, max) {}

template <typename T>
rand_uniform<T>* rand_uniform<T>::clone (void) const
{
	return static_cast<rand_uniform<T>*>(clone_impl());
}

template <typename T>
rand_uniform<T>* rand_uniform<T>::move (void)
{
	return static_cast<rand_uniform<T>*>(move_impl());
}

template <typename T>
itensor_handler<T>* rand_uniform<T>::clone_impl (void) const
{
	return new rand_uniform(*this);
}

template <typename T>
itensor_handler<T>* rand_uniform<T>::move_impl (void)
{
	return new rand_uniform(std::move(*this));
}

template <typename T>
void rand_uniform<T>::calc_data (T* dest, const tensorshape& outshape,
	std::vector<const T*>&, std::vector<tensorshape>&) const
{
	size_t len = outshape.n_elems();
	auto gen = std::bind(distribution_, get_generator());
	std::generate(dest, dest+len, gen);
}

}

#endif