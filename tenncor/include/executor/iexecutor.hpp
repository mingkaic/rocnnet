//
//  iexecutor.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-12.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <stack>
#include "graph/operation/ioperation.hpp"

#pragma once
#ifndef executor_hpp
#define executor_hpp

namespace nnet
{

// does not check if variables in dependencies_ are deleted
// owns nothing in dependencies_
// please don't delete :(
template <typename T>
class iexecutor
{
	protected:
		std::vector<ivariable<T>*> dependencies_;

		virtual iexecutor<T>* clone_impl (void) = 0;

	public:
		virtual ~iexecutor (void) {}

		// COPY
		iexecutor<T>* clone (void);
		
		// MOVE
		
		virtual void add (ivariable<T>* node);

		// stage 2: perform primary objective
		virtual void execute (void) = 0;
		
		// take a snap shot of data before executing (useful for bulk assignment)
		// equivalent to the operation of update (stage 1: perform preliminary actions)
		virtual void freeze (void) = 0;
};

}

#include "../../src/executor/iexecutor.ipp"

#endif /* executor_hpp */
