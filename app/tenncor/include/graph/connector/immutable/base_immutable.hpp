/*!
 *
 *  base_immutable.hpp
 *  cnnet
 *
 *  Purpose:
 *  graph immutable connector interface
 *  manages tensor data and defines abstract
 *  forward computation and backward computation methods
 *
 *  also defines mergable immutable for a
 *  series of forward and backward passes
 *
 *  Created by Mingkai Chen on 2017-06-26.
 *  Copyright © 2017 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/connector/iconnector.hpp"
#include "graph/leaf/variable.hpp"

#pragma once
#ifndef TENNCOR_BASE_IMMUTABLE_HPP
#define TENNCOR_BASE_IMMUTABLE_HPP

namespace nnet
{

template <typename T>
class base_immutable : public iconnector<T>
{
public:
	//! type for mapping leaf nodes to derivative with respect to leaf
	using GRAD_CACHE = std::unordered_map<ileaf<T>*,varptr<T> >;

	virtual ~base_immutable (void);

	// >>>> CLONER & ASSIGNMENT OPERATORS <<<<
	//! clone function
	base_immutable<T>* clone (void) const;

	//! move function
	base_immutable<T>* move (void);

	//! declare copy assignment to copy over transfer functions
	virtual base_immutable<T>& operator = (const base_immutable<T>& other);

	//! declare move assignment to move over transfer functions
	virtual base_immutable<T>& operator = (base_immutable<T>&& other);

	// >>>> FORWARD & BACKWARD DATA <<<<
	//! grab a temporary value traversing top-down
	//! allocates out tensor. caller owns out
	virtual void temporary_eval (const iconnector<T>* target, inode<T>*& out) const;

	//! get gradient wrt some node, applies jacobians before evaluting resulting tensor
	//! may call get_gradient
	virtual varptr<T> derive (inode<T>* wrt);

	//! Utility function: get data shape
	virtual tensorshape get_shape (void) const;

	// >>>> GRAPH STATUS <<<<
	//! get gradient leaves
	virtual std::unordered_set<ileaf<T>*> get_leaves (void) const;

	// >>>> NODE STATUS <<<<
	//! check if the arguments are good; data is available
	virtual bool good_status (void) const;

	//! Inherited from inode: data_ takes data from proto
	virtual bool read_proto (const tenncor::tensor_proto&);

	// >>>> CALLED BY OBSERVER TO UPDATE <<<<
	//! Inherited from iobserver: update data
	//! Updates gcache_ and data_
	virtual void update (std::unordered_set<size_t> argidx);

protected:
	// >>>> CONSTRUCTORS <<<<
	//! base_immutable constructing an aggregate transfer function
	base_immutable (std::vector<inode<T>*> args, std::string label);

	// >>>> COPY && MOVE CONSTRUCTORS <<<<
	//! declare copy constructor to copy over transfer functions
	base_immutable (const base_immutable<T>& other);

	//! declare move constructor to move over transfer functions
	base_immutable (base_immutable<T>&& other);

	// >>>> PROTECTED CLONER <<<<
	//! create a deep copy of this with args
	virtual base_immutable<T>* arg_clone (std::vector<inode<T>*> args) const = 0;

	// >>>> KILL CONDITION <<<<
	//! suicides when all observers die
	void death_on_broken (void);

	// >>>> INTERNAL DATA TRANSFERS <<<<
	//! Forward passing value
	virtual const tensor<T>* get_eval (void) const;

	//! grab operational gradient node, used by other nodes
	//! delay instantiate gcache elements if target leaf was never instantiated
	virtual inode<T>* get_gradient (variable<T>* leaf);

	// >>>> FORWARD & BACKWARD <<<<
	//! forward pass step: populate data_
	virtual void forward_pass (void) = 0;

	//! backward pass step: populate gcache_[leaf]
	virtual void backward_pass (variable<T>* leaf) = 0;

	//! maps leaf to gradient node
	//! lazy instantiates gradient nodes
	//! - stores the gradient value wrt each leaf
	//! - record leaf set
	typename base_immutable<T>::GRAD_CACHE gcache_;

// todo: have an option to disable data_ caching for performance boost
	//! inner tensor to cache forward evaluated values
	tensor<T>* data_ = nullptr;

private:
	//! copy helper
	void copy_helper (const base_immutable& other);

	//! move helper
	void move_helper (base_immutable&& other);

	//! temporary_eval helper
	inode<T>* temp_eval_helper (const iconnector<T>* target, constant<T>*& out) const;
};

}

#include "../../../../src/graph/connector/immutable/base_immutable.ipp"

#endif /* TENNCOR_BASE_IMMUTABLE_HPP */
