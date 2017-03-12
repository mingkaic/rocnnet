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

//! Maps argument data to this data according to shape
template <typename T>
using FORWARD_OP = std::function<void(T*,const tensorshape&,std::vector<const T*>&,std::vector<tensorshape>&)>;

//! Generic Tensor Handler
template <typename T>
class itensor_handler
{
public:
	SHAPER shaper_; //! publicly expose shaper, due its frequent usage

protected:
	//! tensor handler accepts a shape manipulator and a forward transfer function
	itensor_handler (SHAPER shaper, FORWARD_OP<T> forward);

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

	//! performs tensor transfer function given an input array
	void operator () (tensor<T>& out, std::vector<const tensor<T>*> args);
};

//! Constant Initializer
template <typename T>
class const_init : public itensor_handler<T>
{
public:
	//! initialize tensors with a constant scalar value
	const_init (T value);

	//! perform initialization
	void operator () (tensor<T>& out);
};

//! Uniformly Random Initializer
template <typename T>
class rand_uniform : public itensor_handler<T>
{
public:
	//! initialize tensors with a random value between min and max
	rand_uniform (T min, T max);

	//! perform initialization
	void operator () (tensor<T>& out);

private:
	std::uniform_real_distribution<T>  distribution_;
};

} // namespace nnet

#include "../../src/tensor/tensor_handler.ipp"

#endif /* TENNCOR_TENSOR_HANDLER_HPP */
