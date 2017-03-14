/*!
 *
 *  placeholder.hpp
 *  cnnet
 *
 *  Purpose:
 *  placeholder implementation
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include <list>
#include <ctime>
#include <random>
#include <new>
#include <memory>

#include "graph/leaf/ivariable.hpp"

#pragma once
#ifndef TENNCOR_PLACEHOLDER_HPP
#define TENNCOR_PLACEHOLDER_HPP

namespace nnet
{

template <typename T>
class placeholder final : public ivariable<T>
{
public:
	//! shape constructor
	placeholder (const tensorshape& shape, std::string name = "");

	// >>>> CLONE COPY && MOVE <<<<
	//! clone function
	virtual placeholder<T>* clone (void) const;

	//! move constructor
	placeholder (placeholder<T>&& other);

	//! copy assignment
	virtual placeholder<T>& operator = (const placeholder<T>& other);

	//! move assignment
	virtual placeholder<T>& operator = (placeholder<T>&& other);

	// >>>> DATA ASSIGNMENT <<<<
	//! assign raw data according to a
	//! vector representation of inner tensor
	//! for a shape of <d_0, d_1, ..., d_i> and
	//! 	coordinate <c_0, c_1, ..., c_i>:
	//! index mapping function is
	//! sum_j=0:i(product_k=0:j(d_k-1) * c_j) where for k < 0 d_k = 1
	virtual placeholder<T>& operator = (std::vector<T> data);

	//! assign tensor to inner tensor
	virtual placeholder<T>& operator = (tensor<T>& data);

	// >>>> LEAF AND GRADIENT ACCESSORS <<<<
	//! grab operational gradient node, used by other nodes
	virtual inode<T>* get_leaf (variable<T>* leaf) ;

	//! merge/update the gradient/leaf info
	virtual void get_leaves (
		typename inode<T>::GRAD_CACHE& leaves) const;

protected:
	// >>>> CLONE && COPY CONSTRUCTOR <<<<
	//! copy constructor
	placeholder (const placeholder<T>& other);

	//! clone implementation
	virtual inode<T>* clone_impl (void) const;
};

}

#include "../../../src/graph/leaf/placeholder.ipp"

#endif /* TENNCOR_PLACEHOLDER_HPP */
