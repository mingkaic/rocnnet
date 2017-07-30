//
//  alloc_builder.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-03-06.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "memory/alloc_builder.hpp"

#ifdef TENNCOR_ALLOC_BUILDER_HPP

namespace nnet
{

alloc_builder& alloc_builder::get_instance (void)
{
	static alloc_builder singleton;
	return singleton;
}

iallocator* alloc_builder::get (size_t identifier) const
{
	iallocator* al = nullptr;
	auto it = registry_.find(identifier);
	if (registry_.end() != it)
	{
		al = it->second;
	}
	return al;
}

alloc_builder::alloc_builder (void)
{
	// register standard allocators
	registry_[default_alloc::alloc_id] = new default_alloc();
}

alloc_builder::~alloc_builder (void)
{
	// clear registry
	for (auto allocs : registry_)
	{
		delete allocs.second;
	}
}

}

#endif
