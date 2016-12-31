//
//  subject.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/ccoms/ireactive_node.hpp"

#ifdef ireactive_hpp

namespace ccoms
{

bool ireactive_node::safe_destroy (void)
{
	if (suicidal())
	{
		// deletion logic, change here if we allow stack allocation in the future
		delete this;
		return true;
	}
	return false;
}

ireactive_node::~ireactive_node (void)
{
	for (void** ptrr : ptrrs_)
	{
		*ptrr = nullptr;
	}
}

void ireactive_node::set_death (void** ptr)
{
	ptrrs_.emplace(ptr);
}
void ireactive_node::unset_death (void** ptr)
{
	ptrrs_.erase(ptr);
}

}

#endif