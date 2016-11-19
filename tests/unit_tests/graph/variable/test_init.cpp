//
// Created by Mingkai Chen on 2016-08-29.
//

#include "gtest/gtest.h"
#include "graph/initializer.hpp"

// Behavior D000
TEST(INITIALIZER, UnallocTensor_D000)
{
    nnet::random_uniform<double> rinit(-1, 1); 
    nnet::tensor<double> ten; // undefined
    nnet::tensor<double> pten(std::vector<size_t>{0, 1, 2}); // undefined
    EXPECT_DEATH(rinit(ten), ".*");
    EXPECT_DEATH(rinit(pten), ".*");
}