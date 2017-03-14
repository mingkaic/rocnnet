//
// Created by Mingkai Chen on 2017-03-11.
//

//#define DISABLE_MEMORY_MODULE_TESTS
#ifndef DISABLE_MEMORY_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"
#include "fuzz.h"


//#define DISABLE_ALLOCATOR_TEST
#ifndef DISABLE_ALLOCATOR_TEST


// covers allocator
// clone
TEST(ALLOCATOR, Clone_A000)
{}


// covers allocator
// allocate
TEST(ALLOCATOR, Allocate_A001)
{}


// covers allocator
// dealloc
TEST(ALLOCATOR, Deallocate_A002)
{}


// covers allocator
// tracks_size, request_size, alloc_id
TEST(ALLOCATOR, Track_A003)
{}


#endif /* DISABLE_ALLOCATOR_TEST */


#endif /* DISABLE_MEMORY_MODULE_TESTS */
