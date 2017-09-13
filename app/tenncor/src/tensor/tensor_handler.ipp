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
T* itensor_handler<T>::get_raw (tensor<T>& ten) const
{
	assert(ten.is_alloc());
	return ten.raw_data_;
}

template <typename T>
const T* itensor_handler<T>::get_raw (const tensor<T>& ten) const
{
	assert(ten.is_alloc());
	return ten.raw_data_;
}

template <typename T>
shape_extracter<T>::shape_extracter (SHAPE_EXTRACT extract) : shaper_(extract) {}

template <typename T>
shape_extracter<T>* shape_extracter<T>::clone (void) const
{
	return static_cast<shape_extracter<T>*>(this->clone_impl());
}

template <typename T>
shape_extracter<T>* shape_extracter<T>::move (void)
{
	return static_cast<shape_extracter<T>*>(this->move_impl());
}

template <typename T>
void shape_extracter<T>::operator () (tensor<T>& out, std::vector<tensorshape>& ts) const
{
	T* raw = this->get_raw(out);
	std::vector<size_t> shapes = out.get_shape().as_list();
	size_t ncol = shapes[0];
	size_t nrow = shapes.size() < 2 ? 1 : shapes[1];
	std::fill(raw, raw + out.n_elems(), (T) 1);
	for (size_t i = 0; i < nrow; i++)
	{
		std::vector<size_t> sv = shaper_(ts[i]);
		for (size_t j = 0, n = sv.size(); j < n; j++)
		{
			raw[i * ncol + j] = sv[j];
		}
	}
}

template <typename T>
SHAPE_EXTRACT shape_extracter<T>::get_shaper (void) const { return shaper_; }

template <typename T>
itensor_handler<T>* shape_extracter<T>::clone_impl (void) const
{
	return new shape_extracter(*this);
}

template <typename T>
itensor_handler<T>* shape_extracter<T>::move_impl (void)
{
	return new shape_extracter(std::move(*this));
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

template <typename T>
rand_normal<T>::rand_normal (T mean, T stdev) :
	distribution_(mean, stdev) {}

template <typename T>
rand_normal<T>* rand_normal<T>::clone (void) const
{
	return static_cast<rand_normal<T>*>(clone_impl());
}

template <typename T>
rand_normal<T>* rand_normal<T>::move (void)
{
	return static_cast<rand_normal<T>*>(move_impl());
}

template <typename T>
itensor_handler<T>* rand_normal<T>::clone_impl (void) const
{
	return new rand_normal(*this);
}

template <typename T>
itensor_handler<T>* rand_normal<T>::move_impl (void)
{
	return new rand_normal(std::move(*this));
}

template <typename T>
void rand_normal<T>::calc_data (T* dest, tensorshape outshape)
{
	size_t len = outshape.n_elems();
	auto gen = std::bind(distribution_, nnutils::get_generator());
	std::generate(dest, dest+len, gen);
}

}

#endif