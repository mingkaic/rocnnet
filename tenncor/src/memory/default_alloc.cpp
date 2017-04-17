//
//  default_alloc.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-11.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "memory/default_alloc.hpp"

#ifdef TENNCOR_DEFAULT_ALLOC_HPP

namespace nnet
{

const size_t default_alloc::alloc_id = 0;

default_alloc* default_alloc::clone (void) const
{
	return static_cast<default_alloc*>(clone_impl());
}

void* default_alloc::get_raw (size_t, size_t num_bytes)
{
	void* ptr = malloc(num_bytes);
	return ptr;
}

void default_alloc::del_raw (void* ptr, size_t)
{
	free (ptr);
}

iallocator* default_alloc::clone_impl (void) const
{
	return new default_alloc();
}

}

#endif
