//
//  tensor.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2017-02-05.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//
#include <iostream>
#ifdef TENNCOR_TENSOR_HANDLER_HPP

namespace nnet
{

static std::default_random_engine generator(std::time(NULL));

template <typename T>
itensor_handler<T>* itensor_handler<T>::clone (void) const
{
	return clone_impl();
}

template <typename T>
itensor_handler<T>::itensor_handler (itensor_handler<T>&& other) :
	shaper_(std::move(other.shaper_)),
	forward_(std::move(other.forward_)) {}

template <typename T>
itensor_handler<T>& itensor_handler<T>::operator = (const itensor_handler<T>& other)
{
	if (this != &other)
	{
		shaper_ = other.shaper_;
		forward_ = other.forward_;
	}
	return *this;
}

template <typename T>
itensor_handler<T>& itensor_handler<T>::operator = (itensor_handler<T>&& other)
{
	if (this != &other)
	{
		shaper_ = std::move(other.shaper_);
		forward_ = std::move(other.forward_);
	}
	return *this;
}

template <typename T>
itensor_handler<T>::itensor_handler (SHAPER shaper, FORWARD_OP<T> forward) :
	shaper_(shaper), forward_(forward) {}

template <typename T>
itensor_handler<T>::itensor_handler (const itensor_handler<T>& other) :
	shaper_(other.shaper_), forward_(other.forward_) {}

template <typename T>
void itensor_handler<T>::operator () (
	tensor<T>& out,
	std::vector<const tensor<T>*> args)
{
	assert(shaper_ && forward_);
	std::vector<tensorshape> ts;
	std::vector<const T*> raws;
	for (const tensor<T>* arg : args)
	{
		ts.push_back(arg->get_shape());
		raws.push_back(arg->raw_data_);
	}

	tensorshape s = shaper_(ts);
	tensorshape oshape = out.get_shape();
	if (false == s.is_compatible_with(oshape))
	{
		throw std::exception(); // TODO: better exception
	}
	forward_(out.raw_data_, oshape, raws, ts);
}

template <typename T>
transfer_func<T>::transfer_func (SHAPER shaper, FORWARD_OP<T> forward) :
	itensor_handler<T>(shaper, forward) {}

template <typename T>
transfer_func<T>* transfer_func<T>::clone (void) const
{
	return static_cast<transfer_func<T>*>(clone_impl());
}

template <typename T>
transfer_func<T>::transfer_func (transfer_func<T>&& other) :
	itensor_handler<T>(std::move(other)) {}

template <typename T>
transfer_func<T>& transfer_func<T>::operator = (const transfer_func<T>& other)
{
	if (this != &other)
	{
		itensor_handler<T>::operator=(other);
	}
	return *this;
}

template <typename T>
transfer_func<T>& transfer_func<T>::operator = (transfer_func<T>&& other)
{
	if (this != &other)
	{
		itensor_handler<T>::operator = (std::move(other));
	}
	return *this;
}

template <typename T>
void transfer_func<T>::operator () (tensor<T>& out, std::vector<const tensor<T>*> args)
{
	itensor_handler<T>::operator () (out, args);
}

template <typename T>
transfer_func<T>::transfer_func (const transfer_func<T>& other) :
	itensor_handler<T>(other) {}

template <typename T>
itensor_handler<T>* transfer_func<T>::clone_impl (void) const
{
	return new transfer_func(*this);
}

template <typename T>
initializer<T>* initializer<T>::clone (void) const
{
	return static_cast<initializer<T>*>(this->clone_impl());
}

template <typename T>
initializer<T>::initializer (initializer<T>&& other) :
	itensor_handler<T>(std::move(other)) {}

template <typename T>
initializer<T>& initializer<T>::operator = (const initializer<T>& other)
{
	if (this != &other)
	{
		itensor_handler<T>::operator=(other);
	}
	return *this;
}

template <typename T>
initializer<T>& initializer<T>::operator = (initializer<T>&& other)
{
	if (this != &other)
	{
		itensor_handler<T>::operator = (std::move(other));
	}
	return *this;
}

template <typename T>
void initializer<T>::operator () (tensor<T>& out)
{
	itensor_handler<T>::operator ()(out, {});
}

template <typename T>
initializer<T>::initializer (SHAPER shaper, FORWARD_OP<T> forward) :
	itensor_handler<T>(shaper, forward) {}

template <typename T>
initializer<T>::initializer (const initializer<T>& other) :
	itensor_handler<T>(other) {}

template <typename T>
const_init<T>::const_init (T value) :
	initializer<T>(
[](std::vector<tensorshape>) { return tensorshape(); },
[value](T* out, const tensorshape& shape,
	std::vector<const T*>&, std::vector<tensorshape>&)
{
	size_t len = shape.n_elems();
	std::fill(out, out+len, value);
}) {}

template <typename T>
const_init<T>* const_init<T>::clone (void) const
{
	return static_cast<const_init<T>*>(clone_impl());
}

template <typename T>
const_init<T>::const_init (const_init<T>&& other) :
	initializer<T>(std::move(other)) {}

template <typename T>
const_init<T>& const_init<T>::operator = (const const_init<T>& other)
{
	if (this != &other)
	{
		initializer<T>::operator=(other);
	}
	return *this;
}

template <typename T>
const_init<T>& const_init<T>::operator = (const_init<T>&& other)
{
	if (this != &other)
	{
		initializer<T>::operator = (std::move(other));
	}
	return *this;
}

template <typename T>
const_init<T>::const_init (const const_init<T>& other) :
	initializer<T>(other) {}

template <typename T>
itensor_handler<T>* const_init<T>::clone_impl (void) const
{
	return new const_init(*this);
}

template <typename T>
rand_uniform<T>::rand_uniform (T min, T max) :
	distribution_(std::uniform_real_distribution<T>(min, max)),
	initializer<T>(
[](std::vector<tensorshape>) { return tensorshape(); },
[this](T* out, const tensorshape& shape,
	   std::vector<const T*>&, std::vector<tensorshape>&)
{
	size_t len = shape.n_elems();
	for (size_t i = 0; i < len; i++)
	{
		out[i] = distribution_(generator);
	}
}) {}

template <typename T>
rand_uniform<T>* rand_uniform<T>::clone (void) const
{
	return static_cast<rand_uniform<T>*>(clone_impl());
}

template <typename T>
rand_uniform<T>::rand_uniform (rand_uniform<T>&& other) :
	initializer<T>(std::move(other)) {}

template <typename T>
rand_uniform<T>& rand_uniform<T>::operator = (const rand_uniform<T>& other)
{
	if (this != &other)
	{
		initializer<T>::operator=(other);
	}
	return *this;
}

template <typename T>
rand_uniform<T>& rand_uniform<T>::operator = (rand_uniform<T>&& other)
{
	if (this != &other)
	{
		initializer<T>::operator = (std::move(other));
	}
	return *this;
}

template <typename T>
rand_uniform<T>::rand_uniform (const rand_uniform<T>& other) :
	initializer<T>(other) {}

template <typename T>
itensor_handler<T>* rand_uniform<T>::clone_impl (void) const
{
	return new rand_uniform(*this);
}

}

#endif