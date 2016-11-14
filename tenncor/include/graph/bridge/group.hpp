//
//  group.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-30.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef group_hpp
#define group_hpp

#include "iexecutor.hpp"

namespace nnet {

// asynchronously evaluate stored evokers
template <typename T>
class async_group {
	private:
		std::unordered_set<iexecutor<T>*> acts_;

	public:
		void add (iexecutor<T>* exe) {
			acts_.emplace(exe);
		}

		virtual void execute (void) {} // not implemented
};

// sequentially evaluates stored evokers
template <typename T>
class group {
	private:
		std::vector<iexecutor<T>*> acts_;

	public:
		void add (iexecutor<T>* exe) {
			acts_.push_back(exe);
		}

		virtual void execute (void) {
			for (iexecutor<T>* exe : acts_) {
				exe->execute();
			}
		}
};

}

#include "../../../src/graph/bridge/group.ipp"

#endif /* group_hpp */
