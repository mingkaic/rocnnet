//
//  iallocator.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "memory/iallocator.hpp"

#ifdef TENNCOR_ALLOCATOR_HPP

namespace nnet
{
	
iallocator* iallocator::clone (void) const {
	return clone_impl();
}

iallocator* iallocator::move (void)
{
	return move_impl();
}

bool iallocator::tracks_size (void) const { return false; }

size_t iallocator::requested_size (void*) const
{
	throw std::bad_function_call();
	return 0;
}

optional<size_t> iallocator::alloc_id (void*) const { return optional<size_t>(); }

void iallocator::gather_stat (alloc_stat& stats) const { stats.clear(); }


}

#endif
