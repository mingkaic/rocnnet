//
// Created by Mingkai Chen on 2017-03-14.
//

#define DISABLE_GRAPH_MODULE_TESTS
#ifndef DISABLE_GRAPH_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"
#include "fuzz.h"


//#define DISABLE_CONSTANT_TEST
#ifndef DISABLE_CONSTANT_TEST


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
	nnet::inode<double>* gradient = c->get_gradient();
	std::vector<double> res = nnet::expose<double>(gradient);
	ASSERT_EQ(0, res.size());
	delete c;
}


#endif /* DISABLE_CONSTANT_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */
