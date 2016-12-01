//
// Created by Mingkai Chen on 2016-08-29.
//

#include "gtest/gtest.h"
#include "tensor/tensor.hpp"
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


TEST(INITIALIZER, ConstInit)
{
	double constant = rand();
	nnet::const_init<double> cinit(constant);

	nnet::tensor<double> ten1(std::vector<size_t>{1});
	nnet::tensor<double> ten2(std::vector<size_t>({2, 2}));

	cinit(ten1);
	cinit(ten2);

	std::vector<double> raw1 = nnet::expose<double>(&ten1);
	std::vector<double> raw2 = nnet::expose<double>(&ten2);

	EXPECT_EQ(constant, raw1[0]);
	for (double r : raw2)
	{
		EXPECT_EQ(constant, r);
	}
}


TEST(INITIALIZER, RandInt)
{
	nnet::random_uniform<double> rinit(-1, 1);

	nnet::tensor<double> ten1(std::vector<size_t>{1});
	nnet::tensor<double> ten2(std::vector<size_t>{2, 2});

	rinit(ten1);
	rinit(ten2);

	std::vector<double> raw1 = nnet::expose<double>(&ten1);
	std::vector<double> raw2 = nnet::expose<double>(&ten2);

	EXPECT_LT(-1, raw1[0]);
	EXPECT_GT(1, raw1[0]);
	for (double r : raw2)
	{
		EXPECT_LT(-1, r);
		EXPECT_GT(1, r);
	}
}