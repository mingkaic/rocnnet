//
//
//
/*!
 *
 *  shape_dep.hpp
 *  cnnet
 *
 *  Purpose:
 *  Obtain shape information from dependencies
 *  Shape is organized by:
 *  	Each row i represents the info of dependency i
 *  	Each column element represents the dimensional value
 *  Tensor data_ is guaranteed to be 2-D
 *
 *  Created by Mingkai Chen on 2017-07-03.
 *  Copyright © 2017 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/connector/immutable/base_immutable.hpp"

#pragma once
#ifndef TENNCOR_SHAPE_DEP_HPP
#define TENNCOR_SHAPE_DEP_HPP

namespace nnet
{

// todo: make tensor unaligned
template <typename T>
class shape_dep : public base_immutable<T>
{
public:
	virtual ~shape_dep (void);

	// >>>> BUILDER TO FORCE HEAP ALLOCATION <<<<
	//! builder for immutables, grabs ownership of Nf
	static shape_dep<T>* get (std::vector<inode<T>*> args,
		SHAPE_EXTRACT forward, tensorshape shape, std::string name);

	// >>>> CLONER & ASSIGNMENT OPERATORS <<<<
	//! clone function
	shape_dep<T>* clone (void) const;

	//! move function
	shape_dep<T>* move (void);

	//! declare copy assignment to copy over transfer functions
	virtual shape_dep<T>& operator = (const shape_dep<T>& other);

	//! declare move assignment to move over transfer functions
	virtual shape_dep<T>& operator = (shape_dep<T>&& other);

protected:
	// >>>> CONSTRUCTORS <<<<
	//! immutable constructing an aggregate transfer function
	shape_dep (std::vector<inode<T>*> args, SHAPE_EXTRACT forward,
		tensorshape shape, std::string label);

	//! declare copy constructor to copy over transfer functions
	shape_dep (const shape_dep<T>& other);

	//! declare move constructor to move over transfer functions
	shape_dep (shape_dep<T>&& other);

	// >>>> POLYMORPHIC CLONERS <<<<
	//! implement clone function
	virtual inode<T>* clone_impl (void) const;

	//! move implementation
	virtual inode<T>* move_impl (void);

	// >>>> PROTECTED CLONER <<<<
	//! create a deep copy of this with args
	virtual base_immutable<T>* arg_clone (std::vector<inode<T>*> args) const;

	// >>>> FORWARD & BACKWARD <<<<
	//! forward pass step: populate data_
	virtual void forward_pass (void);

	//! backward pass step: populate gcache_[leaf]
	virtual void backward_pass (variable<T>* leaf);

private:
	//! extract shape dimensions to data_
	shape_extracter<T>* shape_info = nullptr;

	tensorshape shape_;
};

}

#include "../../../../src/graph/connector/immutable/shape_dep.ipp"

#endif /* TENNCOR_SHAPE_DEP_HPP */
