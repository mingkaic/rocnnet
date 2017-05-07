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

#include "graph/connector/iconnector.hpp"
#include "graph/leaf/variable.hpp"

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
	//! builder for immutables
	static immutable<T>* get (std::vector<inode<T>*> args,
		SHAPER shaper, FORWARD_OP<T> Nf, BACK_MAP<T> F,
		std::string name, inode<T>* ignore_jacobian = nullptr);

	//! destructor
	virtual ~immutable (void);

	// >>>> CLONE, COPY && MOVE <<<<
	//! clone function
	immutable<T>* clone (void) const;

	//! move function
	immutable<T>* move (void);

	//! declare copy assignment to copy over transfer functions
	virtual immutable<T>& operator = (const immutable<T>& other);

	//! declare move assignment to move over transfer functions
	virtual immutable<T>& operator = (immutable<T>&& other);

	// >>>> ACCESSORS <<<<
	//! Utility function: get data shape
	virtual tensorshape get_shape (void) const;

	//! Forward passing value
	virtual const tensor<T>* get_eval (void) const;

	//! grab a temporary value traversing top-down
	//! allocates out tensor. caller owns out
	virtual void temporary_eval (const iconnector<T>* target, inode<T>*& out) const;

	//! check if the arguments are good; data is available
	virtual bool good_status (void) const;

	//! get gradient leaves
	virtual void get_leaves (
		typename inode<T>::GRAD_CACHE& leaves) const;

	// >>>> MUTATORS
	//! grab operational gradient node, used by other nodes
	//! delay instantiate gcache elements if target leaf was never instantiated
	virtual inode<T>* get_leaf (variable<T>* leaf);

	//! get gradient wrt some node, applies jacobians before evaluting resulting tensor
	//! may call get_leaf
	virtual inode<T>* get_gradient (inode<T>* wrt);

	//! Inherited from iobserver: update data
	//! Updates gcache_ and data_
	virtual void update (subject* arg);

	//! Inherited from inode: data_ takes data from proto
	virtual bool read_proto (const tenncor::tensor_proto&);

protected:
	// >>>> CONSTRUCTORS <<<<
	//! mutable constructor defining transfer functions
	immutable (std::vector<inode<T>*> args,
		SHAPER shaper, FORWARD_OP<T> Nf,
		BACK_MAP<T> F, std::string label);

	//! copy everything but with new arguments
	immutable (std::vector<inode<T>*> args, const immutable<T>& other);

	// >>>> EXECUTE ON KILL CONDITION <<<<
	//! ovride smart destruction,
	//! executed when any dependency is destroyed
	virtual void commit_sudoku (void);

	// >>>> COPY && MOVE CONSTRUCTORS <<<<
	//! implement clone function
	virtual inode<T>* clone_impl (void) const;

	//! move implementation
	virtual inode<T>* move_impl (void);

	//! declare copy constructor to copy over transfer functions
	immutable (const immutable<T>& other);

	//! declare move constructor to move over transfer functions
	immutable (immutable<T>&& other);

	//! list of jacobian transfer function
	//! to be executed on resulting root node
	//! execution order: front to back
	//! insertion order: back to front
	struct JList
	{
		JList (void) :
			uid_(nnutils::uuid(this)) {}

		std::string uid_;
		std::list<JTRANSFER<T> > list_;
	};

	//! jacobian for each variable
	using JCACHE = std::unordered_map<variable<T>*,JList>;
	JCACHE jacobians_;

	std::unique_ptr<constant<T> > zero; //! commonly used zero constant
	std::unique_ptr<constant<T> > one; //! commonly used one constant

private:
	//! initialization helper
	void common (void);

	//! copy helper
	void copy_helper (const immutable& other);

	//! move helper
	void move_helper (immutable&& other);

	// >>>> FORWARD OPERATIONS <<<<
	//! inner tensor to cache forward evaluated values
	std::unique_ptr<tensor<T> > data_ = nullptr;

	//! forward transfer function
	transfer_func<T> Nf_; //! calculates forward passing data

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

#include "../../../../src/graph/connector/immutable/immutable.ipp"

#endif /* TENNCOR_IMMUTABLE_HPP */
