/*!
 *
 *  immutable.hpp
 *  cnnet
 *
 *  Purpose:
 *  graph immutable connector interface
 *  manages connector gradient information
 *
 *  Created by Mingkai Chen on 2017-02-28.
 *  Copyright © 2017 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/operation/iconnector.hpp"

#pragma once
#ifndef TENNCOR_IMMUTABLE_HPP
#define TENNCOR_IMMUTABLE_HPP

namespace nnet
{

//! backward transfer function, get gradient nodes; F: Nf -> Nb
template <typename T>
using BACK_MAP = std::function<inode<T>*(std::vector<inode<T>*>,variable<T>*)>;

//! jacobian transfer function
template <typename T>
using JTRANSFER = std::function<inode<T>*(inode<T>*,variable<T>*)>;

template <typename T>
class immutable : public iconnector<T>
{
public:
	//! destruction
	virtual ~immutable (void) {}

	// >>>> CLONE, COPY && MOVE <<<<
	//! clone function
	immutable<T>* clone (void) const;

	//! declare move constructor to move over transfer functions
	immutable (immutable<T>&& other);

	//! declare copy assignment to copy over transfer functions
	virtual immutable<T>& operator = (const immutable<T>& other);

	//! declare move assignment to move over transfer functions
	virtual immutable<T>& operator = (immutable<T>&& other);

	//! get gradient wrt some node
	virtual const tensor<T>* get_gradient (inode<T>* wrt) const;

	//! check if the arguments are good; data is available
	virtual bool good_status (void) const;

	//! grab operational gradient node, used by other nodes
	virtual inode<T>* get_leaf (variable<T>* leaf);

	//! get gradient leaves
	virtual void get_leaves (
		typename inode<T>::GRAD_CACHE& leaves) const;

protected:
	// >>>> CONSTRUCTORS <<<<
	//! mutable constructor defining transfer functions
	immutable (std::vector<inode<T>*> args,
		BACK_MAP<T> F, std::string name);

	//! list of jacobian transfer function
	//! to be executed on resulting root node
	//! execution order: front to back
	//! insertion order: back to front
	std::list<JTRANSFER<T> > jacobians_;

	// >>>> COPY CONSTRUCTORS <<<<
	//! declare copy constructor to copy over transfer functions
	immutable (const immutable<T>& other);

private:
	void common (void);

	std::unique_ptr<constant<T> > zero; //! commonly used zero constant
	std::unique_ptr<constant<T> > one; //! commonly used one constant

	// >>>> GRAD_ INITIALIZER <<<<
	//! backward transfer function to
	//! lazy instantiate gradient cache values
	BACK_MAP<T> ginit_;

	// >>>> LEAF-GRADIENT CACHE <<<<
	//! maps leaf to gradient node
	//! lazy instantiates gradient nodes
	//! - stores the gradient value wrt each leaf
	//! - record leaf set
	typename inode<T>::GRAD_CACHE gcache_;
};

}

#include "../../../../src/graph/operation/immutable/immutable.ipp"

#endif /* TENNCOR_IMMUTABLE_HPP */
