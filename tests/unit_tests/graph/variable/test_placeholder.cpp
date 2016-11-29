//
// Created by Mingkai Chen on 2016-11-20.
//

#include "gtest/gtest.h"
#include "graph/variable/placeholder.hpp"
#include "mock_operation.h"
using ::testing::_;


// Behavior D100
TEST(PLACEHOLDER, InitNotify_D100)
{
	nnet::placeholder<double> invar(std::vector<size_t>{5, 5}, "const_arr");
	MockOperation* op = MockOperation::build(&invar); // observes cvar
	EXPECT_CALL(*op, update(_)).Times(1);
	// call update on initializations
	std::vector<double> raw;
	for (size_t i = 0; i < 25; i++)
	{
		raw.push_back(rand());
	}
	invar = raw;
	delete op;
}


// Behavior D300
TEST(PLACEHOLDER, Initialization_D300) {
	const size_t insize = 20;
	nnet::placeholder<double> invar((std::vector<size_t>{4, 5}), "in");
	std::vector<double> raw;
	for (size_t i = 0; i < insize; i++) {
		raw.push_back(rand());
	}
	invar = raw;
	std::vector<double> res = nnet::expose<double>(&invar);
	ASSERT_EQ(res.size(), insize);
	for (size_t i = 0; i < insize; i++) {
		EXPECT_EQ(raw[i], res[i]);
	}
}


// Behavior D301
TEST(PLACEHOLDER, ConstructInit_D301) {
	double constant = rand();
	nnet::const_init<double> cinit(constant);
	const size_t insize = 20;
	nnet::placeholder<double> invar((std::vector<size_t>{4, 5}), cinit, "in");
	std::vector<double> res = nnet::expose<double>(&invar);
	ASSERT_EQ(res.size(), insize);
	for (size_t i = 0; i < insize; i++) {
		EXPECT_EQ(constant, res[i]);
	}
}


// Behavior D302
TEST(PLACEHOLDER, ZeroConstGrad_D302) {
	nnet::placeholder<double> invar(std::vector<size_t>{1}, "in");
	invar = std::vector<double>{1};
	nnet::ivariable<double>* gradient = invar.get_gradient();
	std::vector<double> res = nnet::expose<double>(gradient);
	ASSERT_EQ(res.size(), 1);
	EXPECT_EQ(0, res[0]);
}
