//
//  ireactive_node.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-20
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <cassert>
#include <unordered_set>
#include "utils/utils.hpp"

#pragma once
#ifndef ireactive_hpp
#define ireactive_hpp

namespace ccoms
{

// abstract for communication nodes that records leaves
class ireactive_node
{
	private:
		// used exclusively for controlling to pointers of this
		// upon destruction all pointers ptrrs point to will be set to null
		std::unordered_set<void**> ptrrs_;
	
	protected:
		// returns true if suicide on safe_destroy
		// we should always have protected constructors with a static builder
		// if suicidal is true, since suicide never accounts for stack allocation
		virtual bool suicidal (void) = 0;

		ireactive_node (void) {} // just because we need copy constructor
		ireactive_node (ireactive_node& other) {} // prevent ptrrs from being copied
	
	public:
		virtual ~ireactive_node (void); // set all ptrrs' pointer to null
		// return true if this is successfully flagged for deletion
		virtual bool safe_destroy (void);
		// set ptr to null on death,
		// ptr might not necessary point to this, ptr could point to something affecting this
		// this distinction must be determined by the caller, be warned
		void set_death (void** ptr);
		void unset_death (void** ptr);
};

}

#endif /* ireactive_hpp */