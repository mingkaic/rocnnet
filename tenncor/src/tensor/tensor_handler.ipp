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
	std::vector<const tensor<T>*> args)
{
	std::vector<tensorshape> ts;
	std::vector<const T*> raws;
	bool allshaped = true;
	for (const tensor<T>* arg : args)
	{
		if (arg)
		{
			assert(arg->is_alloc());
			ts.push_back(arg->get_shape());
			raws.push_back(arg->raw_data_);
		}
		else
		{
			allshaped = false;
			ts.push_back(tensorshape());
			raws.push_back(nullptr);
		}
	}
	if (allshaped)
	{
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
		}
	}
	assert(out->raw_data_);
	calc_data(out->raw_data_, out->get_shape(), raws, ts);
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
void assign_func<T>::operator () (tensor<T>*& out, const tensor<T>* arg)
{
	itensor_handler<T>::operator () (out, {arg});
}

template <typename T>
void assign_func<T>::operator () (tensor<T>& out, std::vector<T> indata)
{
	tensorshape outshape = out.get_shape();
	assert(indata.size() >= outshape.n_elems());
	std::vector<const T*> in{&indata[0]};
	std::vector<tensorshape> inshapes;
	calc_data(this->get_raw(out), outshape, in, inshapes);
}

template <typename T>
tensorshape assign_func<T>::calc_shape (std::vector<tensorshape> shapes) const
{
	return shapes[0];
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
void assign_func<T>::calc_data (T* dest, const tensorshape& outshape,
	std::vector<const T*>& srcs, std::vector<tensorshape>&)
{
	for (size_t i = 0, n = outshape.n_elems(); i < n; i++)
	{
		dest[i] = f_(dest[i], srcs[0][i]);
	}
}

template <typename T>
transfer_func<T>::transfer_func (SHAPER shaper,
	std::vector<OUT_MAPPER> outidxer,
	ELEM_FUNC<T> aggregate) :
shaper_(shaper),
aggregate_(aggregate),
outidxer_(outidxer) {}

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
void transfer_func<T>::operator () (tensor<T>*& out, std::vector<const tensor<T>*> args)
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
	std::vector<const T*>& srcs, std::vector<tensorshape>& inshapes)
{
	size_t n_args = srcs.size();
	if (n_args == 0) return;
	assert(n_args == inshapes.size());
	assert(n_args == outidxer_.size());
	size_t n_out_elems = outshape.n_elems();
	// only need to execute once
	if (arg_indices_.empty() || false == incache_.is_alloc())
	{
		arg_indices_.clear();
		incache_.deallocate();

		// assert: all inshapes are valid and srcs are non-nullptr
		for (size_t i = 0; i < n_args; i++)
		{
			for (size_t j = 0; j < n_out_elems; j++)
			{
				std::vector<size_t> inidx = outidxer_[i](j, inshapes[i], outshape);
				arg_indices_.insert(arg_indices_.end(), inidx.begin(), inidx.end());
			}
		}
		group_size_ = arg_indices_.size() / (n_args * n_out_elems);
		incache_.allocate(std::vector<size_t>{n_args, group_size_, n_out_elems});
	}
	T* cache_ptr = this->get_raw(incache_);
	size_t idx_block_size = group_size_ * n_out_elems;
	// execute for every dependency update (any non-null src)
	for (size_t i = 0; i < n_args; i++)
	{
		if (const T* src = srcs[i])
		{
			size_t inlimit = inshapes[i].n_elems();
			// pull up indices of argument i
			// we take advantage of tensor's continguous ordering:
			auto argit = arg_indices_.begin() + i * idx_block_size;
			for (size_t j = 0; j < idx_block_size; j++)
			{
				size_t ini = *(argit + j);
				if (ini < inlimit)
				{
					cache_ptr[i + j * n_args] = src[ini];
				}
				else
				{
					cache_ptr[i + j * n_args] = 0;
				}
			}
		}
	}
	size_t data_block_size = n_args * group_size_;
	for (size_t i = 0; i < n_out_elems; i++)
	{
		dest[i] = aggregate_(cache_ptr + i * data_block_size, data_block_size);
	}
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
void initializer<T>::operator () (tensor<T>*& out)
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
	std::vector<const T*>&, std::vector<tensorshape>&)
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
	std::vector<const T*>&, std::vector<tensorshape>&)
{
	size_t len = outshape.n_elems();
	auto gen = std::bind(distribution_, get_generator());
	std::generate(dest, dest+len, gen);
}

}

#endif