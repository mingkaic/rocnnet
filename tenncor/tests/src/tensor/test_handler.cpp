//
// Created by Mingkai Chen on 2017-03-10.
//

#include <algorithm>

#include "gtest/gtest.h"
#include "fuzz.h"


#define DISABLE_HANDLER_TEST
#ifndef DISABLE_HANDLER_TEST


// cover transfer_func, const_init, rand_uniform
// copy constructor and assignment
TEST(HANDLER, Copy_C000)
{}


// cover transfer_func, const_init, rand_uniform
// move constructor and assignment
TEST(HANDLER, Move_C000)
{}


TEST(HANDLER, _C001)
{}


#endif /* DISABLE_HANDLER_TEST */
