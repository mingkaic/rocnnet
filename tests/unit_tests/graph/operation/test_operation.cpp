//
// Created by Mingkai Chen on 2016-08-29.
//

#include "gtest/gtest.h"
#include "mock_operation.h"
#include "graph/functions.hpp"


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
	ASSERT_EQ(sigres, 0.5);

	fanin = std::vector<double>{1};
	sigres = nnet::expose<double>(res)[0];
	double err = 0.73105857863 - sigres;
	ASSERT_LT(err, 0.0001);
}