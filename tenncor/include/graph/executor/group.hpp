//
//  group.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-30.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "iexecutor.hpp"

#pragma once
#ifndef group_hpp
#define group_hpp

namespace nnet
{

// asynchronously evaluate stored evokers
template <typename T>
class async_group
{
	private:
		std::unordered_set<iexecutor<T>*> acts_;

	public:
		void add (iexecutor<T>* exe);
		virtual void execute (void); // not yet implemented
};

// sequentially evaluates stored evokers
template <typename T>
class group {
	private:
		std::vector<std::pair<iexecutor<T>*,bool> > acts_;

	public:
		virtual ~group (void);
	
		// owns determines whether ownership of exe is passed into group
		// having ownership means group is allowed to destroy exe on destruction
		void add (iexecutor<T>* exe, bool owns = false);
		virtual void execute (void);
};

}

#include "../../../src/graph/executor/group.ipp"

#endif /* group_hpp */
