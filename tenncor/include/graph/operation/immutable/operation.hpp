/*!
 *
 *  operation.hpp
 *  cnnet
 *
 *  Purpose:
 *  the immutable operation implementation
 *  manages forward transfer function and tensor
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include <unordered_map>
#include <algorithm>
#include <random>
#include <functional>
#include <limits>
#include <memory>
#include <stack>
#include <new>

#include "tensor/tensor_handler.hpp"
#include "graph/leaf/constant.hpp"
#include "graph/operation/immutable/immutable.hpp"

#pragma once
#ifndef TENNCOR_OPERATION_HPP
#define TENNCOR_OPERATION_HPP

namespace nnet
{

template <typename T>
class operation : public immutable<T>
{
public:
	//! builder for operation
	static operation<T>* get (std::vector<inode<T>*> args,
		SHAPER shaper, FORWARD_OP<T> Nf, BACK_MAP<T> F, std::string name);

	//! destructor
	virtual ~operation (void) {}

	// >>>> CLONE, COPY && MOVE <<<<
	//! Clone function
	operation<T>* clone (void) const;

	//! Move function
	operation<T>* move (void);

	//! Declare copy assignment to copy over transfer functions
	virtual operation<T>& operator = (const operation<T>& other);

	//! Declare move assignment to move over transfer functionsF
	virtual operation<T>& operator = (operation<T>&& other);

	// >>>> ACCESSORS <<<<
	//! Utility function: get data shape
	virtual tensorshape get_shape (void) const;

	//! Forward passing value
	virtual const tensor<T>* get_eval (void) const;

	//! grab a temporary value traversing top-down
	//! allocates out tensor. caller owns out
	virtual void temporary_eval (const iconnector<T>* target,
		tensor<T>*& out) const;

	// >>>> SOLE MUTATOR <<<<
	//! Inherited from iobserver: update data
	//! Updates gcache_ and data_
	virtual void update (subject* arg);

protected:
	// >>>> CONSTRUCTOR <<<<
	//! constructor defining transfer functions
	operation (std::vector<inode<T>*> args,
		SHAPER shaper, FORWARD_OP<T> Nf,
		BACK_MAP<T> F, std::string name);

	// >>>> EXECUTE ON KILL CONDITION <<<<
	//! ovride smart destruction,
	//! executed when any dependency is destroyed
	virtual void commit_sudoku (void);

	// >>>> COPY && MOVE CONSTRUCTOR <<<<
	//! declare copy constructor to copy over transfer functions
	operation (const operation<T>& other);

	//! Declare move constructor to move over transfer functions
	operation (operation<T>&& other);

	//! implement clone function
	virtual inode<T>* clone_impl (void) const;

	//! move implementation
	virtual inode<T>* move_impl (void);

	//! stores whether operation is allocated on heap
	bool onheap_ = false;

private:
	//! inner tensor
	std::unique_ptr<tensor<T> > data_ = nullptr;

	//! forward transfer function
	transfer_func<T> Nf_; //! calculates forward passing data
};

}

#include "../../../../src/graph/operation/immutable/operation.ipp"

#endif /* TENNCOR_OPERATION_HPP */
