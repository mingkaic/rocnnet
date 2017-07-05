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
	// >>>> CONSTRUCTORS <<<<
	//! scalar constructor.
	//! all the benefits of constant, but reassignable
	variable (T scalar, std::string name = "scalar");

	//! shape constructor, initializer is null
	variable (const tensorshape& shape, std::string name = "");

	//! shape constructor with initializer
	variable (const tensorshape& shape,
		const initializer<T>& init, std::string name = "");

	// >>>> CLONER <<<<
	//! clone function
	variable<T>* clone (void) const;

	//! move function
	variable<T>* move (void);

	// >>>> GRAPH STATUS <<<<
	//! merge/update the gradient/leaf info
	virtual void get_leaves (typename inode<T>::GRAD_CACHE& leaves) const;

	// >>>> VARIABLE SPECIAL <<<<
	//! copy over initializer, replace current initializer
	void set_initializer (const initializer<T>& init);

	//! initialize data and returns if possible,
	//! throws error otherwise
	tensor<T>& initialize (void);

	//! initialize data using shape and
	//! returns if possible, throws error otherwise
	tensor<T>& initialize (tensorshape shape);

	//! return update data function (directly assign input node data to this)
	variable_updater<T> assign (inode<T>* input) const;

	//! return update data function (add input node data to this)
	variable_updater<T> assign_add (inode<T>* input) const;

	//! return update data function (subtract input node data to this)
	variable_updater<T> assign_sub (inode<T>* input) const;

	// >>>> TODO: HIDE THIS <<<<
	//! grab operational gradient node, used by other nodes
	virtual void get_leaf (varptr<T>& out, variable<T>* leaf) ;

protected:
	// >>>> POLYMORPHIC CLONERS <<<<
	//! clone implementation
	virtual inode<T>* clone_impl (void) const;

	//! move implementation
	virtual inode<T>* move_impl (void);
};

}

#include "../../../src/graph/leaf/variable.ipp"

#endif /* TENNCOR_VARIABLE_HPP */
