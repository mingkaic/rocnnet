//
// Created by Mingkai Chen on 2017-03-11.
//

#ifndef DISABLE_MEMORY_MODULE_TESTS

#include "gtest/gtest.h"
#include "fuzz.h"

#include "mocks/mock_allocator.h"


#ifndef DISABLE_ALLOCATOR_TEST


// covers allocator
// clone
TEST(ALLOCATOR, Clone_A000)
{
	FUZZ::reset_logger();
	
	// test default allocator
	default_alloc assign;
	default_alloc a;
	iallocator* aptr = &a;
	
	iallocator* cpy = aptr->clone();
	default_alloc* mcpy = dynamic_cast<default_alloc*>(cpy);
	ASSERT_NE(nullptr, mcpy);
	EXPECT_FALSE(mcpy->tracks_size());
	assign = a;
	EXPECT_FALSE(assign.tracks_size());
	delete cpy;
	
	// test tracked allocator 
}


 // covers allocator
 // move
 TEST(ALLOCATOR, Move_A000)
 {
 	FUZZ::reset_logger();
	
 	// test default allocator
 	default_alloc assign;
 	default_alloc a;
 	iallocator* aptr = &a;
	
 	iallocator* mv = aptr->move();
 	default_alloc* mmv = dynamic_cast<default_alloc*>(mv);
 	ASSERT_NE(nullptr, mmv);
 	EXPECT_FALSE(mmv->tracks_size());
 	assign = std::move(a);
 	EXPECT_FALSE(assign.tracks_size());
 	delete mv;
	
 	// test tracked allocator
 }


// covers allocator
// allocate
TEST(ALLOCATOR, Default_Allocate_A001)
{
	FUZZ::reset_logger();
	mock_default_allocator a;

	size_t numa = FUZZ::getInt(1, "numa", {1, 127})[0];
	void* ca = a.template allocate<double>(numa);
	size_t abytes = a.tracker[ca].num_bytes;
	EXPECT_EQ(numa * sizeof(double), abytes);
	free((char*) ca);

	ca = a.template allocate<uint64_t>(numa);
	abytes = a.tracker[ca].num_bytes;
	EXPECT_EQ(numa * sizeof(uint64_t), abytes);
	free((char*) ca);

	ca = a.template allocate<uint32_t>(numa);
	abytes = a.tracker[ca].num_bytes;
	EXPECT_EQ(numa * sizeof(uint32_t), abytes);
	free((char*) ca);

	ca = a.template allocate<uint16_t>(numa);
	abytes = a.tracker[ca].num_bytes;
	EXPECT_EQ(numa * sizeof(uint16_t), abytes);
	free((char*) ca);
}


// allocate an absurdly large amount of memory to cause error
// (line 58, /tenncor/include/memory/iallocator.hpp)
//TEST(ALLOCATOR, Default_LargeAllocate_A001)
//{
//
//}


// covers allocator
// dealloc, dependent on allocate
TEST(ALLOCATOR, Default_Deallocate_A002)
{
	FUZZ::reset_logger();
	mock_default_allocator a;
	size_t numa = FUZZ::getInt(1, "numa", {1, 127})[0];
	size_t numb = FUZZ::getInt(1, "numb", {0, 127})[0];
	void* ca = a.template allocate<uint64_t>(numa);

	// b is not tracked, so 0 should still deallocate
	EXPECT_NE(a.tracker.end(), a.tracker.find(ca));
	a.template dealloc<uint64_t>((uint64_t*) ca, numb);
	EXPECT_EQ(a.tracker.end(), a.tracker.find(ca));
}


// covers allocator
// tracks_size, request_size
TEST(ALLOCATOR, Default_Track_A003)
{
	FUZZ::reset_logger();
	mock_default_allocator a;
	EXPECT_FALSE(a.tracks_size());
	size_t numa = FUZZ::getInt(1, "numa", {1, 127})[0];
	char* ca = (char*)a.template allocate<uint64_t>(numa);
	EXPECT_THROW(a.requested_size(ca), std::bad_function_call);
	a.template dealloc<uint64_t>((uint64_t*) ca, numa);
}


#endif /* DISABLE_ALLOCATOR_TEST */


#endif /* DISABLE_MEMORY_MODULE_TESTS */
