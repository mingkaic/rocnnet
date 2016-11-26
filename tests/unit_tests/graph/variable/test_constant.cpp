//
// Created by Mingkai Chen on 2016-08-29.
//

#include "gtest/gtest.h"
#include "graph/variable/constant.hpp"
#include "mock_operation.h"


// No point in testing Behavior D100 since constant's initialized at construct


// Behavior D400
TEST(CONSTANT, SelfDestruct_D400) {
	nnet::constant<double>* c = nnet::constant<double>::build(1);
	{
 		MockOperation* op = MockOperation::build(c); // observes cvar
	}
	// c is dead. don't delete (if leak, let gtest catch)
}


// Behavior D401
TEST(CONSTANT, Initialization_D401) {
	nnet::constant<double>* c = nnet::constant<double>::build(1);
	EXPECT_TRUE(c->is_init());
	delete c;
}


// Behavior D402
TEST(CONSTANT, ZeroConstGrad_D402) {
	nnet::constant<double>* c = nnet::constant<double>::build(1);
	nnet::ivariable<double>* gradient = c->get_gradient();
	std::vector<double> res = nnet::expose<double>(gradient);
	ASSERT_EQ(res.size(), 1);
	EXPECT_EQ(0, res[0]);
	delete c;
}
