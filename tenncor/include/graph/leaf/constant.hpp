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
	//! override new scalar constructor
	//! in record to determine heap information
	static void* operator new (size_t size, T scalar);

	//! override new raw, shape constructor
	//! in record to determine heap information
	static void* operator new (size_t size, std::vector<T> raw, tensorshape shape);

	//! scalar constructor
	constant (T scalar);

	//! raw and shape constructor
	constant (std::vector<T> raw, tensorshape shape);

	// >>>> CLONE, COPY, & MOVE <<<<
	//! clone function
	virtual constant<T>* clone (void) const;

	//! declare move constructor to prevent onheap from being moved over
	constant (constant<T>&& other);

	//! declare copy assignment to prevent onheap from being copied over
	virtual constant<T>& operator = (const constant<T>& other);

	//! declare move assignment to prevent onheap from being moved over
	virtual constant<T>& operator = (constant<T>&& other);

	// >>>> PUBLICLY ACCESSIBLE GRADIENT <<<<
	//! get gradient wrt some node
	virtual const tensor<T>* get_gradient (inode<T>* wrt) const;

protected:
	const constant<T> zero; //! commonly used constant zero

	// >>>> EXECUTE ON KILL CONDITION <<<<
	//! override smart destruction,
	//! executed when constant loses all audiences,
	//! (after it obtains an audience of course)
	virtual void commit_sudoku (void);

	// >>>> COPY && CLONE <<<<
	//! declare copy constructor to prevent onheap from being copied over
	constant (const constant<T>& other);

	//! clone implementation
	virtual inode<T>* clone_impl (void) const;

	// >>>> LEAF AND GRADIENT ACCESSORS <<<<
	//! merge/update the gradient/leaf info
	virtual void get_leaves (
		typename inode<T>::GRAD_CACHE& leaves) const;

	//! grab operational gradient node, used by other nodes
	virtual inode<T>* get_leaf (variable<T>* leaf) const;

private:
	bool onheap_ = false; //! store heap information
};

}

#include "../../../src/graph/leaf/constant.ipp"

#endif /* TENNCOR_CONSTANT_HPP */