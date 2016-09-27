//
//  allocator.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "../../include/graph/allocator.hpp"

#ifdef allocator_hpp

namespace nnet {

void* iallocator::get_raw (size_t alignment,
    size_t num_bytes) {
    return get_raw(alignment, num_bytes, alloc_attrib());
}

size_t iallocator::requested_size (void* ptr) {
    throw std::bad_function_call();
    return 0;
}

// MEMORY ALLOCATOR IMPLEMENTATION

void* memory_alloc::get_raw (size_t alignment,
    size_t num_bytes,
    alloc_attrib const & attrib) {
    // str8 2 memory, ignore attributes
    void* ptr = malloc(num_bytes);
    return ptr;
}

void memory_alloc::del_raw (void* ptr) {
    // str8 from memory
    free (ptr);
}

memory_alloc* memory_alloc::clone (void) {
    return new memory_alloc();
}

}

#endif
