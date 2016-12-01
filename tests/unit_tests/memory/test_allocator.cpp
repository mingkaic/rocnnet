//
// Created by Mingkai Chen on 2016-11-15.
//


#include "gtest/gtest.h"
#include "memory/ram_alloc.hpp"
#include "tensor/tensor.hpp"

// Behavior B000
TEST(ALLOCATE, FailAllocate_B000)
{
    nnet::tensor<double, nnet::ram_alloc> pure_evil;
    nnet::tensor<double, nnet::ram_alloc> part_evil(std::vector<size_t>{0, 1, 2});
    
    EXPECT_FALSE(pure_evil.is_alloc());
    EXPECT_FALSE(part_evil.is_alloc());
    
    EXPECT_DEATH(pure_evil.allocate(), ".*");
    EXPECT_DEATH(part_evil.allocate(), ".*");
}

// Behavior B001
TEST(ALLOCATE, ReAllocate_B001)
{
    nnet::tensor<double, nnet::ram_alloc> pure_good;
    pure_good.allocate(std::vector<size_t>{1, 2, 3});
    EXPECT_TRUE(pure_good.is_alloc());
}