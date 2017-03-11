/*!
 *
 *  variable.hpp
 *  cnnet
 *
 *  Purpose:
 *  define the graph variable implementation
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
#ifndef TENNCOR_VARIABLE_HPP
#define TENNCOR_VARIABLE_HPP

namespace nnet
{

template <typename T>
class variable final : public ivariable<T>
{
public:
	//! scalar constructor.
	//! all the benefits of constant, but reassignable
	variable (T scalar, std::string name = "scalar");

	//! shape constructor, initializer is null
	variable (const tensorshape& shape, std::string name = "");

	//! shape constructor with initializer
	variable (const tensorshape& shape,
		const itensor_handler<T>& init, std::string name = "");

	// >>>> CLONE COPY && MOVE <<<<
	//! clone function
	virtual variable<T>* clone (void) const;

	//! move constructor
	variable (variable<T>&& other);

	//! copy assignment
	virtual variable<T>& operator = (const variable<T>& other);

	//! move assignment
	virtual variable<T>& operator = (variable<T>&& other);

	// INITIALIZE VALUE
	//! copy over initializer, replace current initializer
	void set_initializer (const itensor_handler<T>& init);

	//! initialize data and returns if possible,
	//! throws error otherwise
	virtual tensor<T>& initialize (void);

	//! initialize data using shape and
	//! returns if possible, throws error otherwise
	virtual tensor<T>& initialize (tensorshape shape);

protected:
	// >>>> CLONE && COPY CONSTRUCTOR <<<<
	//! copy constructor
	variable (const variable<T>& other);

	//! clone implementation
	virtual inode<T>* clone_impl (void) const;

	// >>>> LEAF AND GRADIENT ACCESSORS <<<<
	//! merge/update the gradient/leaf info
	virtual void get_leaves (
		typename inode<T>::GRAD_CACHE& leaves) const;

	//! grab operational gradient node, used by other nodes
	virtual inode<T>* get_leaf (variable<T>* leaf) const;
};

}

#include "../../../src/graph/leaf/variable.ipp"

#endif /* TENNCOR_VARIABLE_HPP */
