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

#include "tensor/tensor.hpp"

#pragma once
#ifndef TENNCOR_TENSOR_HANDLER_HPP
#define TENNCOR_TENSOR_HANDLER_HPP

namespace nnet
{

//! Accumulate an array of shape
using SHAPER = std::function<tensorshape(std::vector<tensorshape>)>;

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
	//! clone function
	itensor_handler<T>* clone (void) const;

	//! explicitly declare move constructor since copy is declared
	itensor_handler (itensor_handler<T>&& other);

	//! copy assignment, since copy and move constructors are explicitly declared
	virtual itensor_handler<T>& operator = (const itensor_handler<T>& other);

	//! move assignment, since copy and move constructors are explicitly declared
	virtual itensor_handler<T>& operator = (itensor_handler<T>&& other);

	SHAPER shaper_; //! publicly expose shaper, due its frequent usage

protected:
	//! tensor handler accepts a shape manipulator and a forward transfer function
	itensor_handler (SHAPER shaper, FORWARD_OP<T> forward);

	//! explicitly declare copy constructor to move from public
	itensor_handler (const itensor_handler<T>& other);

	//! clone implementation
	virtual itensor_handler<T>* clone_impl (void) const = 0;

	//! performs tensor transfer function given an input array
	void operator () (tensor<T>& out, std::vector<const tensor<T>*> args);

private:
	FORWARD_OP<T> forward_; //! raw data transformation function
};

//! Transfer function
template <typename T>
class transfer_func : public itensor_handler<T>
{
public:
	//! tensor handler accepts a shape manipulator and a forward transfer function
	transfer_func (SHAPER shaper, FORWARD_OP<T> forward);

	// >>>> CLONE, MOVE && COPY ASSIGNMENT <<<<
	//! clone function
	transfer_func<T>* clone (void) const;

	//! explicitly declare move constructor since copy is declared
	transfer_func (transfer_func<T>&& other);

	//! copy assignment, since copy and move constructors are explicitly declared
	virtual transfer_func<T>& operator = (const transfer_func<T>& other);

	//! copy assignment, since copy and move constructors are explicitly declared
	virtual transfer_func<T>& operator = (transfer_func<T>&& other);

	//! performs tensor transfer function given an input array
	void operator () (tensor<T>& out, std::vector<const tensor<T>*> args);

protected:
	//! explicitly declare copy constructor to move from public
	transfer_func (const transfer_func<T>& other);

	//! clone implementation
	virtual itensor_handler<T>* clone_impl (void) const;
};

template <typename T>
class initializer : public itensor_handler<T>
{
public:
	// >>>> CLONE, MOVE && COPY ASSIGNMENT <<<<
	//! clone function
	initializer<T>* clone (void) const;

	//! explicitly declare move constructor since copy is declared
	initializer (initializer<T>&& other);

	//! copy assignment, since copy and move constructors are explicitly declared
	virtual initializer<T>& operator = (const initializer<T>& other);

	//! copy assignment, since copy and move constructors are explicitly declared
	virtual initializer<T>& operator = (initializer<T>&& other);

	//! perform initialization
	void operator () (tensor<T>& out);

protected:
	//! initializer forwards functions to itensor_handler
	initializer (SHAPER shaper, FORWARD_OP<T> forward);

	//! explicitly declare copy constructor to move from public
	initializer (const initializer<T>& other);
};

//! Constant Initializer
template <typename T>
class const_init : public initializer<T>
{
public:
	//! initialize tensors with a constant scalar value
	const_init (T value);

	// >>>> CLONE, MOVE && COPY ASSIGNMENT <<<<
	//! clone function
	const_init<T>* clone (void) const;

	//! explicitly declare move constructor since copy is declared
	const_init (const_init<T>&& other);

	//! copy assignment, since copy and move constructors are explicitly declared
	virtual const_init<T>& operator = (const const_init<T>& other);

	//! copy assignment, since copy and move constructors are explicitly declared
	virtual const_init<T>& operator = (const_init<T>&& other);

protected:
	//! explicitly declare copy constructor to move from public
	const_init (const const_init<T>& other);

	//! clone implementation
	virtual itensor_handler<T>* clone_impl (void) const;
};

//! Uniformly Random Initializer
template <typename T>
class rand_uniform : public initializer<T>
{
public:
	//! initialize tensors with a random value between min and max
	rand_uniform (T min, T max);

	// >>>> CLONE, MOVE && COPY ASSIGNMENT <<<<
	//! clone function
	rand_uniform<T>* clone (void) const;

	//! explicitly declare move constructor since copy is declared
	rand_uniform (rand_uniform<T>&& other);

	//! copy assignment, since copy and move constructors are explicitly declared
	virtual rand_uniform<T>& operator = (const rand_uniform<T>& other);

	//! copy assignment, since copy and move constructors are explicitly declared
	virtual rand_uniform<T>& operator = (rand_uniform<T>&& other);

protected:
	//! explicitly declare copy constructor to move from public
	rand_uniform (const rand_uniform<T>& other);

	//! clone implementation
	virtual itensor_handler<T>* clone_impl (void) const;

private:
	std::uniform_real_distribution<T>  distribution_;
};

}

#include "../../src/tensor/tensor_handler.ipp"

#endif /* TENNCOR_TENSOR_HANDLER_HPP */
