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

#include <random>
#include <ctime>

#include "tensor/tensor.hpp"

#pragma once
#ifndef TENNCOR_TENSOR_HANDLER_HPP
#define TENNCOR_TENSOR_HANDLER_HPP

namespace nnet
{

//! Accumulate an array of shape
using SHAPER = std::function<tensorshape(std::vector<tensorshape>)>;

// todo: reduce shape information in forward op (it's unnecessary)
//! Maps argument data to this data according to shape
template <typename T>
using FORWARD_OP = std::function<void(T*,const tensorshape&,std::vector<const T*>&,std::vector<tensorshape>&)>;

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
	void operator () (tensor<T>& out, std::vector<const tensor<T>*> args) const;

	//! calculate raw data from output shape, input shapes, and input data 
	virtual void calc_data (T*,const tensorshape&,
		std::vector<const T*>&,std::vector<tensorshape>&) const = 0;
};

//! Transfer function
template <typename T>
class transfer_func : public itensor_handler<T>
{
public:
	//! tensor handler accepts a shape manipulator and a forward transfer function
	transfer_func (SHAPER shaper, FORWARD_OP<T> forward);

	// >>>> CLONE, MOVE && COPY ASSIGNMENT <<<<
	//! clone function for copying from itensor_handler
	transfer_func<T>* clone (void) const;

	//! clone function for copying from itensor_handler
	transfer_func<T>* move (void);

	//! performs tensor transfer function given an input array
	void operator () (tensor<T>& out, std::vector<const tensor<T>*> args) const;
	
	//! calls shape transformer
	virtual tensorshape calc_shape (std::vector<tensorshape> shapes) const
	{
		return shaper_(shapes);
	}

protected:
	//! clone implementation for copying from itensor_handler
	virtual itensor_handler<T>* clone_impl (void) const;

	//! move implementation for moving from itensor_handler
	virtual itensor_handler<T>* move_impl (void);

	//! calculate raw data using forward functor
	virtual void calc_data (T* dest, const tensorshape& outshape,
		std::vector<const T*>& srcs, std::vector<tensorshape>& inshapes) const
	{
		forward_(dest, outshape, srcs, inshapes);
	}

	SHAPER shaper_; //! shape transformation
	
	FORWARD_OP<T> forward_; //! raw data transformation function
};

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
	void operator () (tensor<T>& out) const;
	
	virtual tensorshape calc_shape (std::vector<tensorshape> shapes) const
	{
		if (shapes.empty()) return {};
		return shapes[0];
	}
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
		std::vector<const T*>&, std::vector<tensorshape>&) const
	{
		size_t len = outshape.n_elems();
		std::fill(dest, dest+len, value_);
	}
		
private:
	T value_;
};

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
		std::vector<const T*>&, std::vector<tensorshape>&) const;

private:
	std::uniform_real_distribution<T>  distribution_;
};

}

#include "../../src/tensor/tensor_handler.ipp"

#endif /* TENNCOR_TENSOR_HANDLER_HPP */
