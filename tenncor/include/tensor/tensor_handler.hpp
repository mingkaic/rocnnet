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

//! Accumulate an array of shape
using SHAPER = std::function<tensorshape(std::vector<tensorshape>)>;

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
		std::function<T(const T&,const T&)> f = [](const T&, const T& src) { return src; }) const;

	//! performs tensor transfer function given an input array
	void operator () (tensor<T>* out, std::vector<T> indata,
		std::function<T(const T&,const T&)> f = [](const T&, const T& src) { return src; }) const;

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
	transfer_func (SHAPER shaper,
		std::vector<OUT_MAPPER> outidxer,
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
	
	//! calls shape transformer
	tensorshape calc_shape (std::vector<tensorshape> shapes) const;

protected:
	//! clone implementation for copying from itensor_handler
	virtual itensor_handler<T>* clone_impl (void) const;

	//! move implementation for moving from itensor_handler
	virtual itensor_handler<T>* move_impl (void);

	SHAPER shaper_; //! shape transformation

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
using general_distribution = std::conditional_t<
	std::is_integral<T>::value, std::uniform_int_distribution<T>,
	std::conditional_t<
		std::is_floating_point<T>::value, std::uniform_real_distribution<T>, void> >;

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
	general_distribution<T>  distribution_;
};

}

#include "../../src/tensor/tensor_handler.ipp"

#endif /* TENNCOR_TENSOR_HANDLER_HPP */
