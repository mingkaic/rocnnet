/*!
 *
 *  constant.hpp
 *  cnnet
 *
 *  Purpose:
 *  constant node
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

#include "graph/leaf/ileaf.hpp"

#pragma once
#ifndef TENNCOR_CONSTANT_HPP
#define TENNCOR_CONSTANT_HPP

namespace nnet
{

template <typename T>
class constant final : public ileaf<T>
{
public:
	//! builder for scalar
	static constant<T>* get (T scalar);

	//! builder for data and shape
	static constant<T>* get (std::vector<T> raw, tensorshape shape);

	//! builder for move
	static constant<T>* get (constant<T>&& other);

	//! destructor to kill zero
	~constant (void);

	// >>>> CLONE, COPY, & MOVE <<<<
	//! clone function
	virtual constant<T>* clone (void) const;

	//! declare copy assignment to prevent onheap from being copied over
	virtual constant<T>& operator = (const constant<T>& other);

	//! declare move assignment to prevent onheap from being moved over
	virtual constant<T>& operator = (constant<T>&& other);

	// >>>> PUBLICLY ACCESSIBLE GRADIENT <<<<
	//! get gradient wrt some node
	virtual const tensor<T>* get_gradient (inode<T>* wrt) const;

	// >>>> LEAF AND GRADIENT ACCESSORS <<<<
	//! grab operational gradient node, used by other nodes
	virtual inode<T>* get_leaf (variable<T>* leaf) ;

	//! merge/update the gradient/leaf info
	virtual void get_leaves (
		typename inode<T>::GRAD_CACHE& leaves) const;

	bool is_managed_ = false; //! if constant is managed, it will not suicide if it lacks audiences

protected:
	constant<T>* zero = nullptr; //! commonly used constant zero

	//! scalar constructor
	constant (T scalar);

	//! raw and shape constructor
	constant (std::vector<T> raw, tensorshape shape);

	// >>>> EXECUTE ON KILL CONDITION <<<<
	//! override smart destruction,
	//! executed when constant loses all audiences,
	//! (after it obtains an audience of course)
	virtual void commit_sudoku (void);

	// >>>> COPY, MOVE && CLONE <<<<
	//! declare copy constructor to prevent onheap from being copied over
	constant (const constant<T>& other);

	//! declare move constructor to prevent onheap from being moved over
	constant (constant<T>&& other);

	//! clone implementation
	virtual inode<T>* clone_impl (void) const;
};

}

#include "../../../src/graph/leaf/constant.ipp"

#endif /* TENNCOR_CONSTANT_HPP */