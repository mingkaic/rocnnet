//
//  tensor.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2017-02-05.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_TENSOR_HANDLER_HPP

#include "utils/utils.hpp"

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
assign_func<T>* assign_func<T>::clone (void) const
{
	return static_cast<assign_func<T>*>(clone_impl());
}

template <typename T>
assign_func<T>* assign_func<T>::move (void)
{
	return static_cast<assign_func<T>*>(move_impl());
}

template <typename T>
void assign_func<T>::operator () (tensor<T>& out, const tensor<T>& arg,
	std::function<T(const T&,const T&)> f) const
{
	tensorshape outshape = out.get_shape();
	size_t n = arg.n_elems();
	assert(n == outshape.n_elems());
	T* dest = this->get_raw(out);
	const T* src = this->get_raw(arg);
	for (size_t i = 0; i < n; i++)
	{
		dest[i] = f(dest[i], src[i]);
	}
}

template <typename T>
void assign_func<T>::operator () (tensor<T>& out, std::vector<T> indata,
	std::function<T(const T&,const T&)> f) const
{
	tensorshape outshape = out.get_shape();
	size_t n = outshape.n_elems();
	assert(n <= indata.size());
	T* dest = this->get_raw(out);
	for (size_t i = 0; i < n; i++)
	{
		dest[i] = f(dest[i], indata[i]);
	}
}

template <typename T>
itensor_handler<T>* assign_func<T>::clone_impl (void) const
{
	return new assign_func<T>(*this);
}

template <typename T>
itensor_handler<T>* assign_func<T>::move_impl (void)
{
	return new assign_func<T>(std::move(*this));
}

template <typename T>
transfer_func<T>::transfer_func (TRANSFER_FUNC<T> transfer) : transfer_(transfer) {}

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
void transfer_func<T>::operator () (tensor<T>& out, std::vector<const tensor<T>*>& args)
{
	size_t n_arg = args.size();
	T* dest = this->get_raw(out);
	std::vector<const T*> sources(n_arg, nullptr);
	shape_io sio{out.get_shape(), {}};

	std::transform(args.begin(), args.end(), sources.begin(),
	[this, &sio](const tensor<T>* arg) -> const T*
	{
		sio.ins_.push_back(arg->get_shape());
		return this->get_raw(*arg);
	});
	this->transfer_(dest, sources, sio);
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
void initializer<T>::operator () (tensor<T>& out)
{
	this->calc_data(this->get_raw(out), out.get_shape());
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
void const_init<T>::calc_data (T* dest, tensorshape outshape)
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
void rand_uniform<T>::calc_data (T* dest, tensorshape outshape)
{
	size_t len = outshape.n_elems();
	auto gen = std::bind(distribution_, nnutils::get_generator());
	std::generate(dest, dest+len, gen);
}

}

#endif