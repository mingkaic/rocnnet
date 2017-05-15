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

#include "graph/leaf/ivariable.hpp"

#pragma once
#ifndef TENNCOR_VARIABLE_HPP
#define TENNCOR_VARIABLE_HPP

#include <list>
#include <new>
#include <memory>

namespace nnet
{

template <typename T>
using variable_updater = std::function<void(void)>;

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
		const initializer<T>& init, std::string name = "");

	// >>>> CLONE COPY && MOVE <<<<
	//! clone function
	variable<T>* clone (void) const;

	//! move function
	variable<T>* move (void);

	// INITIALIZE VALUE
	//! copy over initializer, replace current initializer
	void set_initializer (const initializer<T>& init);

	//! initialize data and returns if possible,
	//! throws error otherwise
	virtual tensor<T>& initialize (void);

	//! initialize data using shape and
	//! returns if possible, throws error otherwise
	virtual tensor<T>& initialize (tensorshape shape);

	// >>>> LEAF AND GRADIENT ACCESSORS <<<<
	//! grab operational gradient node, used by other nodes
	virtual inode<T>* get_leaf (variable<T>* leaf) ;

	//! merge/update the gradient/leaf info
	virtual void get_leaves (
		typename inode<T>::GRAD_CACHE& leaves) const;

	variable_updater<T> assign (inode<T>* input) const;

	//! return update data function (add input node data to this)
	variable_updater<T> assign_add (inode<T>* input) const;

	//! return update data function (add input node data to this)
	variable_updater<T> assign_sub (inode<T>* input) const;

protected:
	//! clone implementation
	virtual inode<T>* clone_impl (void) const;

	//! move implementation
	virtual inode<T>* move_impl (void);
};

}

#include "../../../src/graph/leaf/variable.ipp"

#endif /* TENNCOR_VARIABLE_HPP */
