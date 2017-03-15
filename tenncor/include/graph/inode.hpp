/*!
 *
 *  inode.hpp
 *  cnnet
 *
 *  Purpose:
 *  graph variable interface
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include "memory/session.hpp"
#include "tensor/tensor_handler.hpp"
#include "tensor/tensor.hpp"
#include "utils/utils.hpp"
#include "graph/react/subject.hpp"

#pragma once
#ifndef TENNCOR_INODE_HPP
#define TENNCOR_INODE_HPP

namespace nnet
{

// TODO: limit T down to numeric types using c++17 std::variant

template <typename T>
class ioptimizer;

template <typename T>
class variable;

template <typename T>
class iconnector;

template <typename T>
class inode : public react::subject
{
public:
	//! unregister from session TODO: deprecate session, find a better way of mass data serialization
	virtual ~inode (void);

	// >>>> CLONE, COPY && MOVE ASSIGNMENTS <<<<
	//! clone function
	inode<T>* clone (void) const;

	//! declare copy assignment to prevent id_ copy over
	virtual inode<T>& operator = (const inode<T>& other);

	//! declare move assignment to prevent id_ copy over
	virtual inode<T>& operator = (inode<T>&& other);

	// >>>> ACCESSORS <<<<
	//! get the unique hash value
	std::string get_uid (void) const;

	//! Get the non-unique label
	std::string get_label (void) const { return name_; }

	//! get a pretty and unique label
	virtual std::string get_name (void) const;

	//! utility function: get data shape
	virtual tensorshape get_shape (void) const = 0;

	//! check if data is available
	virtual bool good_status (void) const = 0;

	//! get forward passing value
	virtual const tensor<T>* get_eval (void) const = 0;

	//! get top-level gradient value, used by root nodes
	virtual const tensor<T>* get_gradient (inode<T>* wrt) const = 0;

	//! Grab operational gradient node, used by other nodes
	virtual inode<T>* get_leaf (variable<T>* leaf)  = 0;

	//! store the gradient operation wrt to a leaf
	using GRAD_CACHE = std::unordered_map<variable<T>*,inode<T>*>;

	// >>>> META-DATA ACCESSOR <<<<
	//! Merge/Update the gradient/leaf info
	virtual void get_leaves (GRAD_CACHE& leaves) const = 0;

protected:
	// >>>> CONSTRUCTORS <<<<
	//! Default constructor
	inode (std::string name);

	// >>>> COPY && MOVE CONSTRUCTORS <<<<
	//! Declare copy constructor to prevent id_ copy over
	inode (const inode<T>& other);

	//! Declare move constructor to prevent id_ copy over
	inode (inode<T>&& other);

	//! clone abstraction function
	virtual inode<T>* clone_impl (void) const = 0;

	// >>>> NODE META DATA <<<<
	const std::string id_ = nnutils::uuid(this); //! unique hash

private:
	std::string name_; //! variable label
};

//! add helper function for exposing node's data
template <typename T>
std::vector<T> expose (const inode<T>* var);

//! equality check for node against scalars
template <typename T>
bool operator == (const inode<T>& c, T scalar);

//! inequality check for node against scalars
template <typename T>
bool operator != (const inode<T>& c, T scalar);

}

#include "../../src/graph/inode.ipp"

#endif /* TENNCOR_INODE_HPP */