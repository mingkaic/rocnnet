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
void assign_func<T>::operator () (tensor<T>* out, const tensor<T>* arg,
	std::function<T(const T&,const T&)> f) const
{
	tensorshape outshape = out->get_shape();
	size_t n = arg->n_elems();
	assert(n == outshape.n_elems());
	T* dest = this->get_raw(out);
	const T* src = this->get_raw(arg);
	for (size_t i = 0; i < n; i++)
	{
		dest[i] = f(dest[i], src[i]);
	}
}

template <typename T>
void assign_func<T>::operator () (tensor<T>* out, std::vector<T> indata,
	std::function<T(const T&,const T&)> f) const
{
	tensorshape outshape = out->get_shape();
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
transfer_func<T>::transfer_func (std::vector<OUT_MAPPER> outidxer, ELEM_FUNC<T> aggregate) :
	aggregate_(aggregate),
	outidxer_(outidxer) {}

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
void transfer_func<T>::operator () (tensor<T>* out, std::vector<const T*>& args)
{
	size_t n_out = out->n_elems();
	size_t ptr_block_size = args.size() / n_out;
	T* dest = this->get_raw(out);
	for (size_t i = 0; i < n_out; i++)
	{
		dest[i] = aggregate_(&args[i * ptr_block_size], ptr_block_size);
	}
}

template <typename T>
void transfer_func<T>::operator () (std::vector<T>& out, std::vector<const T*>& args)
{
	size_t n_out = out.size();
	size_t ptr_block_size = args.size() / n_out;
	for (size_t i = 0; i < n_out; i++)
	{
		out[i] = aggregate_(&args[i * ptr_block_size], ptr_block_size);
	}
}

template <typename T>
std::vector<const T*> transfer_func<T>::prepare_args (tensorshape outshape,
	std::vector<const tensor<T>*> args) const
{
	size_t n_args = args.size();
	if (n_args == 0) return {};
	assert(n_args == outidxer_.size());
	size_t n_out = outshape.n_elems();
	// rank 0: group
	// rank 1: elements
	// rank 2: arguments
	size_t max_blocksize = 0;
	std::vector<std::vector<size_t> > arg_indices(n_args);
	for (size_t i = 0; i < n_args; i++)
	{
		if (args[i])
		{
			tensorshape ashape = args[i]->get_shape();
			std::vector<size_t> arg_index;
			for (size_t j = 0; j < n_out; j++)
			{
				std::vector<size_t> elem_idx = outidxer_[i](j, ashape, outshape);
				arg_index.insert(arg_index.end(), elem_idx.begin(), elem_idx.end());
			}
			arg_indices[i] = arg_index;
			max_blocksize = std::max(max_blocksize, arg_index.size());
		}
	}
	std::vector<const T*> arg(n_args * max_blocksize, nullptr);
	for (size_t i = 0; i < n_args; i++)
	{
		if (args[i])
		{
			size_t n = args[i]->n_elems();
			const T* src = this->get_raw(args[i]);
			for (size_t j = 0; j < arg_indices[i].size(); j++)
			{
				if (arg_indices[i][j] < n) arg[i + j * n_args] = src + arg_indices[i][j];
			}
		}
	}
	return arg;
}

template <typename T>
std::vector<const T*> transfer_func<T>::prepare_args (tensorshape outshape,
	std::vector<std::pair<T*,tensorshape> > args) const
{
	size_t n_args = args.size();
	if (n_args == 0) return {};
	assert(n_args == outidxer_.size());
	size_t n_out = outshape.n_elems();
	// rank 0: group
	// rank 1: elements
	// rank 2: arguments
	size_t max_blocksize = 0;
	std::vector<std::vector<size_t> > arg_indices(n_args);
	for (size_t i = 0; i < n_args; i++)
	{
		if (args[i].first)
		{
			tensorshape ashape = args[i].second;
			std::vector<size_t> arg_index;
			for (size_t j = 0; j < n_out; j++)
			{
				std::vector<size_t> elem_idx = outidxer_[i](j, ashape, outshape);
				arg_index.insert(arg_index.end(), elem_idx.begin(), elem_idx.end());
			}
			arg_indices[i] = arg_index;
			max_blocksize = std::max(max_blocksize, arg_index.size());
		}
	}
	std::vector<const T*> arg(n_args * max_blocksize, nullptr);
	for (size_t i = 0; i < n_args; i++)
	{
		if (args[i].first)
		{
			const T* src = args[i].first;
			size_t n = args[i].second.n_elems();
			for (size_t j = 0; j < arg_indices[i].size(); j++)
			{
				if (arg_indices[i][j] < n)
					arg[i + j * n_args] = src + arg_indices[i][j];
			}
		}
	}
	return arg;
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
void initializer<T>::operator () (tensor<T>* out)
{
	this->calc_data(this->get_raw(out), out->get_shape());
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