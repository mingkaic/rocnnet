//
//  iexecutor.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-12.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef executor_hpp
#define executor_hpp

#include <stack>
#include "graph/operation/ioperation.hpp"

namespace nnet {

// dangerous, sources might be deleted, and this wouldn't be notified (wise to make observer at some point...)
// not observer yet because observer REQUIRES subject at construction
// alternatively, check session registry before doing anything with srcs_
template <typename T>
class iexecutor {
	protected:
		std::vector<ivariable<T>*> srcs_;

		void copy (const iexecutor<T>& other) { srcs_ = other.srcs_; }
		virtual iexecutor<T>* clone_impl (void) = 0;

	public:
		iexecutor<T>* clone (void) { return clone_impl(); }

		void add (ivariable<T>* node) { srcs_.push_back(node); }

		virtual void execute (void) = 0;
};

}

#include "../../../src/graph/bridge/iexecutor.ipp"

#endif /* executor_hpp */
