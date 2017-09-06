/*!
 *
 *  tensor_handler.hpp
 *  cnnet
 *
 *  Purpose:
 *  handler is a delegate for manipulating raw datas in tensors
 *
 *  Created by Mingkai Chen on 2017-02-05.
 *  Copyright © 2017 Mingkai Chen. All rights reserved.
 *
 */

#include "tensor/tensor.hpp"

#pragma once
#ifndef TENNCOR_TENSOR_HANDLER_HPP
#define TENNCOR_TENSOR_HANDLER_HPP

namespace nnet
{

using SHAPE_EXTRACT = std::function<std::vector<size_t>(tensorshape&)>;

using OUT_MAPPER = std::function<std::vector<size_t>(size_t,tensorshape&,const tensorshape&)>;

template <typename T>
using ELEM_FUNC = std::function<T(const T**,size_t)>;

//! Generic Tensor Handler
template <typename T>
class itensor_handler
{
public:
	//! virtual handler interface destructor
	virtual ~itensor_handler (void) {}

	// >>>> CLONE, MOVE && COPY ASSIGNMENT <<<<
	//! clone function for copying from itensor_handler
	itensor_handler<T>* clone (void) const;

	//! clone function for copying from itensor_handler
	itensor_handler<T>* move (void);

protected:
	//! clone implementation for copying from itensor_handler
	virtual itensor_handler<T>* clone_impl (void) const = 0;

	//! move implementation for moving from itensor_handler
	virtual itensor_handler<T>* move_impl (void) = 0;

	T* get_raw (tensor<T>* ten) const
	{
		assert(ten->is_alloc());
		return ten->raw_data_;
	}

	const T* get_raw (const tensor<T>* ten) const
	{
		assert(ten->is_alloc());
		return ten->raw_data_;
	}
};

template <typename T>
class shape_extracter : public itensor_handler<T>
{
public:
	shape_extracter (SHAPE_EXTRACT extract) : shaper_(extract) {}

	// >>>> CLONE, MOVE && COPY ASSIGNMENT <<<<
	//! clone function for copying from itensor_handler
	shape_extracter<T>* clone (void) const
	{
		return static_cast<shape_extracter<T>*>(this->clone_impl());
	}

	//! clone function for copying from itensor_handler
	shape_extracter<T>* move (void)
	{
		return static_cast<shape_extracter<T>*>(this->move_impl());
	}

	//! extract tensor information and store in out
	void operator () (tensor<T>* out, std::vector<tensorshape>& ts) const
	{
		T* raw = this->get_raw(out);
		std::vector<size_t> shapes = out->get_shape().as_list();
		size_t ncol = shapes[0];
		size_t nrow = shapes.size() < 2 ? 1 : shapes[1];
		std::fill(raw, raw+out->n_elems(), (T) 1);
		for (size_t i = 0; i < nrow; i++)
		{
			std::vector<size_t> sv = shaper_(ts[i]);
			for (size_t j = 0, n = sv.size(); j < n; j++)
			{
				raw[i * ncol + j] = sv[j];
			}
		}
	}

protected:
	//! clone implementation for copying from itensor_handler
	virtual itensor_handler<T>* clone_impl (void) const
	{
		return new shape_extracter(*this);
	}

	//! move implementation for moving from itensor_handler
	virtual itensor_handler<T>* move_impl (void)
	{
		return new shape_extracter(std::move(*this));
	}

private:
	//! extract shape dimensions to data_
	SHAPE_EXTRACT shaper_;
};

template <typename T>
T default_assign (const T&, const T& src) { return src; }

// todo: test
//! assigns one tensor to another
template <typename T>
class assign_func : public itensor_handler<T>
{
public:
	// >>>> CLONE, MOVE && COPY ASSIGNMENT <<<<
	//! clone function for copying from itensor_handler
	assign_func<T>* clone (void) const;

	//! clone function for copying from itensor_handler
	assign_func<T>* move (void);

	//! performs tensor transfer function given an input tensor
	void operator () (tensor<T>* out, const tensor<T>* arg,
		std::function<T(const T&,const T&)> f = default_assign<T>) const;

	//! performs tensor transfer function given an input array
	void operator () (tensor<T>* out, std::vector<T> indata,
		std::function<T(const T&,const T&)> f = default_assign<T>) const;

protected:
	//! clone implementation for copying from itensor_handler
	virtual itensor_handler<T>* clone_impl (void) const;

	//! move implementation for moving from itensor_handler
	virtual itensor_handler<T>* move_impl (void);
};

//! Transfer Function
template <typename T>
class transfer_func : public itensor_handler<T>
{
public:
	//! tensor handler accepts a shape manipulator and a forward transfer function
	transfer_func (std::vector<OUT_MAPPER> outidxer,
		ELEM_FUNC<T> aggregate);

	// >>>> CLONE, MOVE && COPY ASSIGNMENT <<<<
	//! clone function for copying from itensor_handler
	transfer_func<T>* clone (void) const;

	//! clone function for copying from itensor_handler
	transfer_func<T>* move (void);

	//! performs tensor transfer function given an input cache
	void operator () (tensor<T>* out, std::vector<const T*>& args);

	// todo: test
	void operator () (std::vector<T>& out, std::vector<const T*>& args);

	// only need to execute once per immutable
	std::vector<const T*> prepare_args (tensorshape outshape,
		std::vector<const tensor<T>*> args) const;

	// todo: test
	// mimicks preparation from tensors
	std::vector<const T*> prepare_args (tensorshape outshape,
		std::vector<std::pair<T*,tensorshape> > args) const;

protected:
	//! clone implementation for copying from itensor_handler
	virtual itensor_handler<T>* clone_impl (void) const;

	//! move implementation for moving from itensor_handler
	virtual itensor_handler<T>* move_impl (void);

	//! element-wise aggregate elements
	ELEM_FUNC<T> aggregate_;

// todo: move out to immutable, since handlers should not say WHAT and WHERE the data is, just HOW it is calculated
	//! element-wise reference input from output (only used once when populating new tensor)
	std::vector<OUT_MAPPER> outidxer_;
};

//! Initializer Handler
template <typename T>
class initializer : public itensor_handler<T>
{
public:
	// >>>> CLONE, MOVE && COPY ASSIGNMENT <<<<
	//! clone function for copying from initializer
	initializer<T>* clone (void) const;

	//! clone function for copying from initializer
	initializer<T>* move (void);

	//! perform initialization
	void operator () (tensor<T>* out);

protected:
	virtual void calc_data (T* dest, tensorshape outshape) = 0;
};

//! Constant Initializer
template <typename T>
class const_init : public initializer<T>
{
public:
	//! initialize tensors with a constant scalar value
	const_init (T value);

	//! clone function for copying from parents
	const_init<T>* clone (void) const;

	//! clone function for copying from parents
	const_init<T>* move (void);

protected:
	//! clone implementation for copying from parents
	virtual itensor_handler<T>* clone_impl (void) const;

	//! move implementation for moving from parents
	virtual itensor_handler<T>* move_impl (void);
	
	//! initialize data as constant
	virtual void calc_data (T* dest, tensorshape outshape);
		
private:
	T value_;
};

template <typename T>
using general_uniform = std::conditional_t<
	std::is_integral<T>::value, std::uniform_int_distribution<T>,
	std::conditional_t<std::is_floating_point<T>::value,
		std::uniform_real_distribution<T>, void> >;

//! Uniformly Random Initializer
template <typename T>
class rand_uniform : public initializer<T>
{
public:
	//! initialize tensors with a random value between min and max
	rand_uniform (T min, T max);

	//! clone function for copying from parents
	rand_uniform<T>* clone (void) const;

	//! clone function for copying from parents
	rand_uniform<T>* move (void);

protected:
	//! clone implementation for copying from parents
	virtual itensor_handler<T>* clone_impl (void) const;

	//! clone function for copying from parents
	virtual itensor_handler<T>* move_impl (void);
	
	//! initialize data as constant
	virtual void calc_data (T* dest, tensorshape outshape);

private:
	general_uniform<T>  distribution_;
};

//! Normal Random Initializer
template <typename T>
class rand_normal : public initializer<T>
{
public:
	//! initialize tensors with a random value between min and max
	rand_normal (T mean = 0, T stdev = 1) :
		distribution_(mean, stdev) {}

	//! clone function for copying from parents
	rand_normal<T>* clone (void) const
	{
		return static_cast<rand_normal<T>*>(clone_impl());
	}

	//! clone function for copying from parents
	rand_normal<T>* move (void)
	{
		return static_cast<rand_normal<T>*>(move_impl());
	}

protected:
	//! clone implementation for copying from parents
	virtual itensor_handler<T>* clone_impl (void) const
	{
		return new rand_normal(*this);
	}

	//! clone function for copying from parents
	virtual itensor_handler<T>* move_impl (void)
	{
		return new rand_normal(std::move(*this));
	}

	//! initialize data as constant
	virtual void calc_data (T* dest, tensorshape outshape)
	{
		size_t len = outshape.n_elems();
		auto gen = std::bind(distribution_, nnutils::get_generator());
		std::generate(dest, dest+len, gen);
	}

private:
	std::normal_distribution<T> distribution_;
};

}

#include "../../src/tensor/tensor_handler.ipp"

#endif /* TENNCOR_TENSOR_HANDLER_HPP */
