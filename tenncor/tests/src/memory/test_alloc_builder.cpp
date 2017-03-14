//
// Created by Mingkai Chen on 2017-03-11.
//

//#define DISABLE_MEMORY_MODULE_TESTS
#ifndef DISABLE_MEMORY_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"
#include "fuzz.h"


//#define DISABLE_ALLOC_BUILDER_TEST
#ifndef DISABLE_ALLOC_BUILDER_TEST


// covers alloc_builder
TEST(ALLOC_BUILDER, Singleton_B000)
{}


// covers allocator
// allocate
TEST(ALLOC_BUILDER, Allocate_A001)
{}


// covers allocator
// dealloc
TEST(ALLOC_BUILDER, Deallocate_A002)
{}


// covers allocator
// tracks_size, request_size, alloc_id
TEST(ALLOC_BUILDER, Track_A003)
{}


#endif /* DISABLE_ALLOC_BUILDER_TEST */


#endif /* DISABLE_MEMORY_MODULE_TESTS */
