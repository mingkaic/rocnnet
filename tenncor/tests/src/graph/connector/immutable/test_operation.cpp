//
// Created by Mingkai Chen on 2016-08-29.
//

#define DISABLE_GRAPH_MODULE_TESTS
#ifndef DISABLE_GRAPH_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"
#include "fuzz.h"


//#define DISABLE_OPERATION_TEST
#ifndef DISABLE_OPERATION_TEST


TEST(OPERATION, dot)
{

}


TEST(OPERATION, high_dim_mul)
{

}


TEST(OPERATION, univar_func)
{
	nnet::placeholder<double> fanin((std::vector<size_t>{1}), "in");
	nnet::varptr<double> res = nnet::sigmoid<double>(&fanin);

	fanin = std::vector<double>{0};

	double sigres = nnet::expose<double>(res)[0];
	EXPECT_EQ(sigres, 0.5);

	fanin = std::vector<double>{1};
	sigres = nnet::expose<double>(res)[0];
	double err = 0.73105857863 - sigres;
	EXPECT_LT(err, 0.0001);
}


#endif /* DISABLE_OPERATION_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */