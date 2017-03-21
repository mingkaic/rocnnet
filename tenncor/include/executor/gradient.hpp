/*!
 *
 *  gradient.hpp
 *  cnnet
 *
 *  Purpose:
 *  computes the gradient of a root node
 *  depends on the root node
 *
 *  Created by Mingkai Chen on 2016-11-12.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include "iexecutor.hpp"
#include "graph/leaf/constant.hpp"

#define gradient_hpp
#pragma once
#ifndef gradient_hpp
#define gradient_hpp

namespace nnet
{
	
template <typename T>
using GRAD_GATHER = std::function<void(inode<T>*,placeholder<T>*)>;

template <typename T>
class gradient : public iexecutor<T>
{
public:
	//! calculate the first derivative value of root
	//! if leaf is nullptr, derive wrt all leaves
	//! otherwise, derive wrt leaf
	//! freeze by default
	gradient (inode<T>* root,
		inode<T>* leaf = nullptr)
	{
		this->add_dependency(root);
		this->freeze();
	}

	//! calculate the first derivative wrt all leaves
	//! except for the variables specified in ignore
	gradient (inode<T>* root,
		std::unordered_set<variable<T>*> ignore) :
	gradient<T>(root)
	{
		// remove all ignore that are not leaves of root

		ignore_ = ignore;
	}

	// >>>> CLONE, COPY && MOVE ASSIGNMENT <<<<
	//! clone function
	gradient<T>* clone (void) const
	{
		return static_cast<gradient<T>*>(clone_impl());
	}

	//! move constructor to move ignore set

	//! copy assignment to copy ignore set
	gradient<T>& operator = (const gradient<T>& other);

	void collect_grad (GRAD_GATHER<T> collector);

protected:
	//! copy constructor to copy over ignore
	iexecutor (const iexecutor<T>& other)
	{
	}

	//! clone implementation
	virtual iexecutor<T>* clone_impl (void) const
	{
		return new iexecutor(*this);
	}

	//! execute node
	virtual void executor (inode<T>* n)
	{

	}


	void clear_map (void);

	void copy (const gradient<T>& other);
	gradient (const gradient<T>& other);

private:
	// id to bind leaf_map_
	const std::string gid_ = nnutils::uuid(this);

	// predefine leaf of gradient connector
	std::vector<inode<T>*> potential_srcs_;

	// graph data (root, and leaf)
	inode<T>* g_root_;
	GRAD_MAP<T> leaf_map_;
};

}

#include "../../src/executor/gradient.ipp"

#endif /* gradient_hpp */
