//
// Created by Mingkai Chen on 2017-03-11.
//

#ifndef DISABLE_MEMORY_MODULE_TESTS

#include <algorithm>
#include <thread>
#include <mocks/mock_allocator.h>

#include "gtest/gtest.h"
#include "fuzz.h"
#include "mocks/mock_allocator.h"

#include "memory/default_alloc.hpp"
#include "memory/alloc_builder.hpp"


using namespace nnet;


//#define DISABLE_BUILDER_TEST
#ifndef DISABLE_BUILDER_TEST


static const size_t GETTER_ID = 1523;
static const size_t SETTER_ID = 2352932;
static size_t SETTER_ITER = 0;
static const size_t CHECK_ID = 111112;


// covers alloc_builder
// singleton property
TEST(ALLOC_BUILDER, Singleton_B000)
{
	FUZZ::delim();
	// same thread
	alloc_builder& inst1 = alloc_builder::get_instance();
	alloc_builder& inst2 = alloc_builder::get_instance();

	void* i3ptr = nullptr;
	// different thread
	std::thread other ([&i3ptr](void) {
		alloc_builder& inst = alloc_builder::get_instance();
		i3ptr = &inst;
	});

	other.join(); // wait to avoid using ipc

	ASSERT_EQ(&inst1, &inst2);
	ASSERT_EQ(i3ptr, &inst2);
}


// covers allocator
// registertype
TEST(ALLOC_BUILDER, Register_A001)
{
	FUZZ::delim();
	const size_t realsetter = SETTER_ID + SETTER_ITER;
	alloc_builder& builder = alloc_builder::get_instance();

	EXPECT_TRUE(builder.template registertype<mock_allocator>(realsetter));

	// registering a used type is accepted
	EXPECT_TRUE(builder.template registertype<mock_allocator>(realsetter+1));
	EXPECT_TRUE(builder.template registertype<default_alloc>(realsetter+2));

	// registering the a used id (regardless of type) will return false
	EXPECT_FALSE(builder.registertype<mock_allocator>(realsetter));
	EXPECT_FALSE(builder.registertype<default_alloc>(realsetter));
	SETTER_ITER+=3;
}


// covers allocator
// get, depends on registertype
TEST(ALLOC_BUILDER, Get_A002)
{
	FUZZ::delim();
	alloc_builder& builder = alloc_builder::get_instance();

	builder.registertype<mock_allocator>(GETTER_ID);
	// getting invalid allocator
	assert(156 < GETTER_ID/2);
	size_t randid = FUZZ::getInt(1, {156, GETTER_ID/2})[0];
	EXPECT_EQ(nullptr, builder.get(randid));
	// getting internal allocator
	iallocator* def = builder.get(default_alloc::alloc_id);
	EXPECT_NE(nullptr, dynamic_cast<default_alloc*>(def));
	// getting custom allocator
	def = builder.get(GETTER_ID);
	mock_allocator* mock = dynamic_cast<mock_allocator*>(def);
	EXPECT_NE(nullptr, mock);
	mock->tracksize_ = true;
	size_t id = mock->uid = FUZZ::getInt(1)[0];

	def = builder.get(GETTER_ID);
	mock_allocator* mock2 = dynamic_cast<mock_allocator*>(def);
	EXPECT_TRUE(mock2->tracksize_);
	EXPECT_EQ(id, mock2->uid);
}


// covers allocator
// check registry
TEST(ALLOC_BUILDER, Check_A003)
{
	FUZZ::delim();
	alloc_builder& builder = alloc_builder::get_instance();

	builder.registertype<mock_allocator>(CHECK_ID);
	builder.registertype<default_alloc>(CHECK_ID+1);

	EXPECT_TRUE(builder.template check_registry<mock_allocator>(CHECK_ID));
	EXPECT_TRUE(builder.template check_registry<default_alloc>(CHECK_ID+1));

	EXPECT_FALSE(builder.template check_registry<default_alloc>(CHECK_ID));
	EXPECT_FALSE(builder.template check_registry<mock_allocator>(CHECK_ID+1));
}


#endif /* DISABLE_BUILDER_TEST */


#endif /* DISABLE_MEMORY_MODULE_TESTS */
