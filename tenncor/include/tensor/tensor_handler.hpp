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
using ELEM_FUNC = std::function<T(const T*,size_t)>;

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
	
	//! calculate the shape given input shapes
	//! publicly expose shape calculator due its utility
	virtual tensorshape calc_shape (std::vector<tensorshape> shapes) const = 0;

protected:
	//! clone implementation for copying from itensor_handler
	virtual itensor_handler<T>* clone_impl (void) const = 0;

	//! move implementation for moving from itensor_handler
	virtual itensor_handler<T>* move_impl (void) = 0;

	//! performs tensor transfer function given an input array
	void operator () (tensor<T>*& out, std::vector<const tensor<T>*> args);

	//! calculate raw data from output shape, input shapes, and input data 
	virtual void calc_data (T*,const tensorshape&,
		std::vector<const T*>&,std::vector<tensorshape>&) = 0;

	T* get_raw (tensor<T>& ten) { return ten.raw_data_; }
};

// todo: test
//! assigns one tensor to another
template <typename T>
class assign_func : public itensor_handler<T>
{
public:
	assign_func (void) :
		f_([](const T&, const T& src) { return src; }) {}

	//! assign using some function
	assign_func (std::function<T(const T&,const T&)> f) : f_(f) {}

	// >>>> CLONE, MOVE && COPY ASSIGNMENT <<<<
	//! clone function for copying from itensor_handler
	assign_func<T>* clone (void) const;

	//! clone function for copying from itensor_handler
	assign_func<T>* move (void);

	//! performs tensor transfer function given an input tensor
	void operator () (tensor<T>*& out, const tensor<T>* arg);

	//! performs tensor transfer function given an input array
	void operator () (tensor<T>& out, std::vector<T> indata);

	//! calls shape transformer
	virtual tensorshape calc_shape (std::vector<tensorshape> shapes) const;

protected:
	//! clone implementation for copying from itensor_handler
	virtual itensor_handler<T>* clone_impl (void) const;

	//! move implementation for moving from itensor_handler
	virtual itensor_handler<T>* move_impl (void);

	//! calculate raw data using forward functor
	virtual void calc_data (T* dest, const tensorshape& outshape,
		std::vector<const T*>& srcs, std::vector<tensorshape>&);

private:
	std::function<T(const T&,const T&)> f_;
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

	//! performs tensor transfer function given an input array
	void operator () (tensor<T>*& out, std::vector<const tensor<T>*> args);
	
	//! calls shape transformer
	virtual tensorshape calc_shape (std::vector<tensorshape> shapes) const;

protected:
	//! clone implementation for copying from itensor_handler
	virtual itensor_handler<T>* clone_impl (void) const;

	//! move implementation for moving from itensor_handler
	virtual itensor_handler<T>* move_impl (void);

	//! calculate raw data using forward functor
	virtual void calc_data (T* dest, const tensorshape& outshape,
		std::vector<const T*>& srcs, std::vector<tensorshape>& inshapes);

	SHAPER shaper_; //! shape transformation

	//! element-wise aggregate elements
	ELEM_FUNC<T> aggregate_;

	//! element-wise reference input from output (only used once when populating new tensor)
	std::vector<OUT_MAPPER> outidxer_;

	//! a simulated 3-D tensor of indices:
	//! rank 0: groups
	//! rank 1: elements
	//! rank 2: arguments
	std::vector<size_t> arg_indices_;

	//! groups: rank 0 of arg_indices_, rank 1 of incache_
	size_t group_size_ = 1;

	//! 3-D tensor of temporary data:
	//! rank 0: arguments
	//! rank 1: groups
	//! rank 2: elements
	tensor<T> incache_;
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
	void operator () (tensor<T>*& out);
	
	virtual tensorshape calc_shape (std::vector<tensorshape> shapes) const;
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
	virtual void calc_data (T* dest, const tensorshape& outshape,
		std::vector<const T*>&, std::vector<tensorshape>&);
		
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
	virtual void calc_data (T* dest, const tensorshape& outshape,
		std::vector<const T*>&, std::vector<tensorshape>&);

private:
	general_distribution<T>  distribution_;
};

}

#include "../../src/tensor/tensor_handler.ipp"

#endif /* TENNCOR_TENSOR_HANDLER_HPP */
