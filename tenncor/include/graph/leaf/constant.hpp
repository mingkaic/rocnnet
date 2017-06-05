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

#include "graph/leaf/ileaf.hpp"

#pragma once
#ifndef TENNCOR_CONSTANT_HPP
#define TENNCOR_CONSTANT_HPP

#include <list>
#include <new>
#include <memory>

namespace nnet
{

template <typename T>
class constant final : public ileaf<T>
{
public:
	//! get shared zero constant that is managed
	static constant<T>* get_shared_zero (void);
	
	//! get shared one constant that is managed
	static constant<T>* get_shared_one (void);
	
	//! builder for scalar
	static constant<T>* get (T scalar);

	//! builder for data and shape
	static constant<T>* get (std::vector<T> raw, tensorshape shape);

	// >>>> CLONE, COPY, & MOVE <<<<
	//! clone function
	constant<T>* clone (void) const;

	//! copy function
	constant<T>* move (void);

	// >>>> COPY, MOVE && CLONE <<<<
	//! deleted copy constructor
	constant (const constant<T>& other) = delete;

	//! deleted move constructor
	constant (constant<T>&& other) = delete;

	//! copy assignment deleted
	constant<T>& operator = (const constant<T>& other) = delete;

	//! move assignment deleted
	constant<T>& operator = (constant<T>&& other) = delete;

	// >>>> PUBLICLY ACCESSIBLE GRADIENT <<<<
	//! get gradient wrt some node
	virtual varptr<T> get_gradient (inode<T>* wrt);

	// >>>> LEAF AND GRADIENT ACCESSORS <<<<
	//! grab operational gradient node, used by other nodes
	virtual void get_leaf (inode<T>*& out, variable<T>* leaf) ;

	//! merge/update the gradient/leaf info
	virtual void get_leaves (
		typename inode<T>::GRAD_CACHE& leaves) const;

	bool is_managed_ = false; //! if constant is managed, it will not suicide if it lacks audiences

protected:
	//! scalar constructor
	constant (T scalar);

	//! raw and shape constructor
	constant (std::vector<T> raw, tensorshape shape);

	// >>>> EXECUTE ON KILL CONDITION <<<<
	//! override smart destruction,
	//! executed when constant loses all audiences,
	//! (after it obtains an audience of course)
	virtual void commit_sudoku_sub (void);

	//! clone implementation
	virtual inode<T>* clone_impl (void) const;

	//! move implementation
	virtual inode<T>* move_impl (void);
	
	static constant<T> shared_zero;
	
	static constant<T> shared_one;
};

}

#include "../../../src/graph/leaf/constant.ipp"

#endif /* TENNCOR_CONSTANT_HPP */