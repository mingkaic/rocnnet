//
// Created by Mingkai Chen on 2016-08-29.
//

#include "gtest/gtest.h"
#include "mock_operation.h"
#include "graph/function.hpp"


TEST(OPERATION, dot)
{

}


TEST(OPERATION, high_dim_mul)
{

}


TEST(OPERATION, univar_func)
{
	nnet::placeptr<double> fanin = new nnet::placeholder<double>((std::vector<size_t>{1}), "in");
	nnet::varptr<double> res = nnet::sigmoid<double>(fanin);
	nnet::expose<double>* out = new nnet::expose<double>(res);

	*fanin = std::vector<double>{0};
	double sigres = out->get_raw()[0];
	ASSERT_EQ(sigres, 0.5);

	*fanin = std::vector<double>{1};
	sigres = out->get_raw()[0];
	double err = 0.73105857863 - sigres;
	ASSERT_LT(err, 0.0001);
}