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
	// >>>> BUILDER TO FORCE HEAP ALLOCATION <<<<
	//! get shared zero constant that is managed
	static constant<T>* get_shared_zero (void);
	
	//! get shared one constant that is managed
	static constant<T>* get_shared_one (void);
	
	//! builder for scalar
	static constant<T>* get (T scalar);

	//! builder for data and shape
	static constant<T>* get (std::vector<T> raw, tensorshape shape);

	// >>>> CAN'T COPY OR MOVE (GOES AGAINST SHARING) <<<<
	//! deleted copy constructor
	constant (const constant<T>& other) = delete;

	//! deleted move constructor
	constant (constant<T>&& other) = delete;

	//! copy assignment deleted
	constant<T>& operator = (const constant<T>& other) = delete;

	//! move assignment deleted
	constant<T>& operator = (constant<T>&& other) = delete;

	// >>>> BACKWARD DATA <<<<
	//! get gradient wrt some node
	virtual varptr<T> get_gradient (inode<T>* wrt);

	// >>>> GRAPH STATUS <<<<
	//! merge/update the gradient/leaf info
	virtual void get_leaves (typename inode<T>::GRAD_CACHE& leaves) const;

	// >>>> NODE STATUS <<<<
	//! set this constant as being managed by some node
	//! this will not die if it loses all observers
	void be_managed (void);

	// >>>> TODO: HIDE THIS <<<<
	//! grab operational gradient node, used by other nodes
	virtual void get_leaf (varptr<T>& out, variable<T>* leaf) ;

protected:
	//! scalar constructor
	constant (T scalar);

	//! raw and shape constructor
	constant (std::vector<T> raw, tensorshape shape);

	// >>>> KILL CONDITION <<<<
	//! suicides when this loses all observers (unless this is_managed)
	virtual void death_on_noparent (void);

	// >>>> POLYMORPHIC CLONERS (RETURN NULLS) <<<<
	//! clone implementation
	virtual inode<T>* clone_impl (void) const;

	//! move implementation
	virtual inode<T>* move_impl (void);

private:
	//! commonly used constant: 0
	static constant<T> shared_zero;

	//! commonly used constant: 1
	static constant<T> shared_one;

	//! if constant is managed by some node,
	//! that node is responsible for this node's life cycle
	bool is_managed_ = false;
};

}

#include "../../../src/graph/leaf/constant.ipp"

#endif /* TENNCOR_CONSTANT_HPP */