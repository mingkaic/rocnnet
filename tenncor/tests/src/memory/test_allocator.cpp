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
	FUZZ::delim();
	mock_allocator a(true);
	mock_allocator b(false);
	iallocator* aptr = &a;
	iallocator* bptr = &b;
	iallocator* cpy = aptr->clone();
	iallocator* cpyb = bptr->clone();
	mock_allocator* mcpy = dynamic_cast<mock_allocator*>(cpy);
	mock_allocator* mcpyb = dynamic_cast<mock_allocator*>(cpyb);
	ASSERT_NE(nullptr, mcpy);
	EXPECT_TRUE(mcpy->tracksize_);
	EXPECT_FALSE(mcpyb->tracksize_);

	delete cpy;
	delete cpyb;
}


// covers allocator
// allocate
TEST(ALLOCATOR, Allocate_A001)
{
	FUZZ::delim();
	mock_allocator a(true);
	mock_allocator b(false);

	size_t numa = FUZZ::getInt(1, "numa", {1, 127})[0];
	size_t numb = FUZZ::getInt(1, "numb", {1, 127})[0];
	char* ca = (char*)a.template allocate<double>(numa);
	char* cb = (char*)b.template allocate<double>(numb);
	size_t abytes = a.tracker[ca].num_bytes;
	size_t bbytes = b.tracker[cb].num_bytes;
	EXPECT_EQ(numa * sizeof(double), abytes);
	EXPECT_EQ(numb * sizeof(double), bbytes);
	delete ca;
	delete cb;

	ca = (char*)a.template allocate<uint64_t>(numa);
	cb = (char*)b.template allocate<uint64_t>(numb);
	abytes = a.tracker[ca].num_bytes;
	bbytes = b.tracker[cb].num_bytes;
	EXPECT_EQ(numa * sizeof(uint64_t), abytes);
	EXPECT_EQ(numb * sizeof(uint64_t), bbytes);
	delete ca;
	delete cb;

	ca = (char*)a.template allocate<uint32_t>(numa);
	cb = (char*)b.template allocate<uint32_t>(numb);
	abytes = a.tracker[ca].num_bytes;
	bbytes = b.tracker[cb].num_bytes;
	EXPECT_EQ(numa * sizeof(uint32_t), abytes);
	EXPECT_EQ(numb * sizeof(uint32_t), bbytes);
	delete ca;
	delete cb;

	ca = (char*)a.template allocate<uint16_t>(numa);
	cb = (char*)b.template allocate<uint16_t>(numb);
	abytes = a.tracker[ca].num_bytes;
	bbytes = b.tracker[cb].num_bytes;
	EXPECT_EQ(numa * sizeof(uint16_t), abytes);
	EXPECT_EQ(numb * sizeof(uint16_t), bbytes);
	delete ca;
	delete cb;
}


// allocate an absurdly large amount of memory to cause error
// (line 58, /tenncor/include/memory/iallocator.hpp)
//TEST(ALLOCATOR, LargeAllocate_A001)
//{
//
//}


// covers allocator
// dealloc, dependent on allocate
TEST(ALLOCATOR, Deallocate_A002)
{
	FUZZ::delim();
	mock_allocator a(false);
	mock_allocator b(true);
	mock_allocator c(true);

	size_t numa = FUZZ::getInt(1, "numa", {1, 127})[0];
	size_t numb = FUZZ::getInt(1, "numb", {1, 127})[0];
	size_t numc = FUZZ::getInt(1, "numc", {25, 127})[0];

	char* ca = (char*)a.template allocate<uint64_t>(numa);
	char* cb = (char*)b.template allocate<uint64_t>(numb);
	char* cc = (char*)c.template allocate<uint64_t>(numc);

	// b is not tracked, so 0 should still deallocate
	a.template dealloc<uint64_t>((uint64_t*) ca, 0);
	EXPECT_EQ(a.tracker.end(), a.tracker.find(ca));

	// b is a tracked, deallocate all at once
	b.template dealloc<uint64_t>((uint64_t*) cb, numb);

	size_t i = 0;
	while (i < numc)
	{
		size_t size = 1;
		if (i < numc-1)
		{
			size_t potential = numc-i;
			size = potential;
			if (potential > 5)
			{
				size = FUZZ::getInt(1, "size", {1, potential/2})[0];
			}
		}

		c.template dealloc<uint64_t>(((uint64_t *) cc) + i, size);
		i += size;
	}
}


// covers allocator
// tracks_size, request_size
TEST(ALLOCATOR, Track_A003)
{
	FUZZ::delim();
	mock_allocator a(false);
	mock_allocator b(true);

	EXPECT_FALSE(a.tracks_size());
	EXPECT_TRUE(b.tracks_size());

	size_t numa = FUZZ::getInt(1, "numa", {1, 127})[0];
	size_t numb = FUZZ::getInt(1, "numb", {1, 127})[0];
	char* ca = (char*)a.template allocate<uint64_t>(numa);
	char* cb = (char*)b.template allocate<uint64_t>(numb);

	EXPECT_THROW(a.requested_size(ca), std::bad_function_call);
	ASSERT_EQ(numb * sizeof(uint64_t), b.requested_size(cb));
}


#endif /* DISABLE_ALLOCATOR_TEST */


#endif /* DISABLE_MEMORY_MODULE_TESTS */
