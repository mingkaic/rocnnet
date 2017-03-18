/*!
 *
 *  iexecutor.hpp
 *  cnnet
 *
 *  Purpose:
 *  iexecutor provides an interface
 *  to process nodes
 *
 *  Created by Mingkai Chen on 2016-11-12.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include <stack>

//#include "graph/connector/immutable/operation.hpp"

#define TENNCOR_EXECUTOR_HPP

#pragma once
#ifndef TENNCOR_EXECUTOR_HPP
#define TENNCOR_EXECUTOR_HPP

namespace nnet
{

template <typename T>
class iexecutor : public iobserver
{
public:
	//! virtual destructor
	virtual ~iexecutor (void) {}

	// >>>> CLONE <<<<
	//! clone function
	iexecutor<T>* clone (void) const
	{ return clone_impl(); }

	// >>>> MUTATOR <<<<
	//! stop listening to updates
	void freeze (void) { listen_ = false; }

	//! start listening to updates
	void unfreeze (void)
	{
		listen_ = true;
		for (subject* arg : this->dependencies_)
		{
			executor(static_cast<inode<T>*>(arg));
		}
	}

	//! update observer value according to subject
	virtual void update (subject* arg) final
	{
		if (listen_)
		{
			execute(static_cast<inode<T>*>(arg));
		}
	}

protected:
	//! clone implementation
	virtual iexecutor<T>* clone_impl (void) const = 0;

	//! execute node
	virtual void executor (inode<T>* n) = 0;

private:
	bool listen_ = true;
};

}

#include "../../src/executor/iexecutor.ipp"

#endif /* TENNCOR_EXECUTOR_HPP */
