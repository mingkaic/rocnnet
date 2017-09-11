/*!
 *
 *  immutable.hpp
 *  cnnet
 *
 *  Purpose:
 *  graph immutable connector that manages a
 *  single operator's forward and backward pass
 *
 *  Created by Mingkai Chen on 2017-02-28.
 *  Copyright © 2017 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/connector/immutable/base_immutable.hpp"
#include <memory>

#pragma once
#ifndef TENNCOR_IMMUTABLE_HPP
#define TENNCOR_IMMUTABLE_HPP

namespace nnet
{

template <typename T>
class immutable : public base_immutable<T>
{
public:
	virtual ~immutable (void);

	// >>>> BUILDER TO FORCE HEAP ALLOCATION <<<<
	//! builder for immutables, grabs ownership of Nf
	static immutable<T>* get (std::vector<inode<T>*> args,
		SHAPER shaper, transfer_func<T>* Nf,
		BACK_MAP<T> ginit, std::string name,
		inode<T>* ignore_jacobian = nullptr);

	// >>>> CLONER & ASSIGNMENT OPERATORS <<<<
	//! clone function
	immutable<T>* clone (void) const;

	//! move function
	immutable<T>* move (void);

	//! declare copy assignment to copy over transfer functions
	virtual immutable<T>& operator = (const immutable<T>& other);

	//! declare move assignment to move over transfer functions
	virtual immutable<T>& operator = (immutable<T>&& other);

	// >>>> GRAPH STATUS <<<<
	//! summarize this immutable
	virtual typename iconnector<T>::summary_series summarize (void) const;

protected:
	// >>>> CONSTRUCTORS <<<<
	//! immutable constructing an aggregate transfer function
	immutable (std::vector<inode<T>*> args,
		SHAPER shaper, transfer_func<T>* Nf,
		BACK_MAP<T> ginit, std::string label);

	//! declare copy constructor to copy over transfer functions
	immutable (const immutable<T>& other);

	//! declare move constructor to move over transfer functions
	immutable (immutable<T>&& other);

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
	//! copy helper
	void copy_helper (const immutable& other);

	//! move helper
	void move_helper (immutable&& other);

	//! calculates shape of this node
	SHAPER shaper_;

	//! forward transfer function
	//! calculates forward passing data
	transfer_func<T>* Nf_ = nullptr;

	//! backward transfer function to
	//! lazy instantiate gradient cache values
	BACK_MAP<T> ginit_;
};

}

#include "../../../../src/graph/connector/immutable/immutable.ipp"

#endif /* TENNCOR_IMMUTABLE_HPP */
