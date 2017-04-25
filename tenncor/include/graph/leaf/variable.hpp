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

	//! return update data function (add input node data to this)
	variable_updater<T> assign_add (inode<T>* input) const
	{
		return [this, input]()
		{
			tensor<T>* outputt = this->data_.get();
			const tensor<T>* inputt = input->get_eval();
			transfer_func<T> assign(
				[outputt](std::vector<tensorshape>)
				{
					return outputt->get_shape();
				},
				[](T* dest, const tensorshape& shape, std::vector<const T*>& srcs, std::vector<tensorshape>& inshapes)
				{
					size_t ns = shape.n_elems();
					tensorshape& ins = inshapes.at(0);
					assert((ins.is_fully_defined() && ns == ins.n_elems()) ||
						(!ins.is_fully_defined() && 0 == ns % ins.n_known()));
					assert(nullptr != srcs[0]);
					for (size_t i = 0; i < ns; i++)
					{
						dest[i] += srcs[0][i];
					}
				});
			assign(*outputt, {inputt});
		};
	}

protected:
	//! clone implementation
	virtual inode<T>* clone_impl (void) const;

	//! move implementation
	virtual inode<T>* move_impl (void);
};

}

#include "../../../src/graph/leaf/variable.ipp"

#endif /* TENNCOR_VARIABLE_HPP */
