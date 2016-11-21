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

// inheriting from iobserver for the sole purpose of deletion once sources (dependencies) are deleted
// could be dangerous if iexecutor is not meant to be destroyed... consider revise later.
template <typename T>
class iexecutor : public ccoms::iobserver // TODO: we really don't need to inherit from iobserver at all...
{
	protected:
		virtual iexecutor<T>* clone_impl (void) = 0;

		iexecutor (void) :
			ccoms::iobserver(std::vector<ccoms::subject*>{}) {}

	public:
		virtual ~iexecutor (void) {}

		// COPY
		iexecutor<T>* clone (void);
		
		// MOVE
		
		virtual void add (ivariable<T>* node);
		
		virtual void update (ccoms::subject* sub) {}
		// stage 2: perform primary objective
		virtual void execute (void) = 0;
		
		// take a snap shot of data before executing (useful for bulk assignment)
		// equivalent to the operation of update (stage 1: perform preliminary actions)
		virtual void freeze (void) = 0;
};

}

#include "../../src/executor/iexecutor.ipp"

#endif /* executor_hpp */
