//
//  allocator.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-11.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "memory/ram_alloc.hpp"

#ifdef ram_alloc_hpp

namespace nnet {

#include "memory/ram_alloc.hpp"


// MEMORY ALLOCATOR IMPLEMENTATION

void* ram_alloc::get_raw (size_t alignment,
						  size_t num_bytes,
						  const alloc_attrib& attrib) {
	// str8 2 memory, ignore attributes
	void* ptr = malloc(num_bytes);
	return ptr;
}

void ram_alloc::del_raw (void* ptr) {
	// str8 from memory
	free (ptr);
}

iallocator* ram_alloc::clone_impl (void) {
	return new ram_alloc();
}

}

#endif