//
// Created by Mingkai Chen on 2016-08-29.
//

#include "gtest/gtest.h"
#include "memory/session.hpp"
#include "graph/variable/variable.hpp"


nnet::const_init<double> cinit(rand());
	

// Behavior D100

// Behavior D200
TEST(VARIABLE, ScalarInit_D200)
{
    
}

TEST(VARIABLE, ConstInit)
{
	nnet::variable<double> cvar(std::vector<size_t>{3, 3}, cinit, "constant_arr");

	sess.initialize_all<double>();
	std::vector<double> raw = nnet::expose<double>(cvar);

	ASSERT_EQ(raw.size(), 9);

	for (double elem : raw) {
		EXPECT_EQ(elem, constant);
	}
}

// Behavior D201
TEST(VARIABLE, Gradient_D201)
{
	nnet::variable<double> cvar(std::vector<size_t>{3, 3}, cinit, "constant_arr");
	ivariable<double>* grad = cvar.get_gradient();
	ASSERT_EQ(&cvar, grad);
}

//
//
//TEST(VARIABLE, RandomInit) {
//	nnet::session& sess = nnet::session::get_instance();
//	nnet::random_uniform<double> init(0.0, 1.0);
//
//	nnet::varptr<double> rvar = new nnet::variable<double>(std::vector<size_t>{5, 5}, init, "random_arr");
//
//	sess.initialize_all<double>();
//	std::vector<double> raw = nnet::expose<double>(rvar);
//
//	ASSERT_EQ(raw.size(), 25);
//
//	double sum = 0;
//	for (double elem : raw) {
//		sum += elem;
//		EXPECT_GT(elem, 0);
//		EXPECT_LT(elem, 1);
//	}
//	ASSERT_GT(sum/raw.size(), 0);
//}
//
//
//TEST(VARIABLE, Placeholder) {
//	const size_t insize = 20;
//	nnet::placeptr<double> invar = new nnet::placeholder<double>((std::vector<size_t>{1, insize}), "in");
//	std::vector<double> sample;
//	for (size_t i = 0; i < insize; i++) {
//		sample.push_back(rand());
//	}
//	*invar = sample;
//
//	std::vector<double> raw = nnet::expose<double>(invar);
//
//	ASSERT_EQ(raw.size(), insize);
//
//	for (size_t i = 0; i < insize; i++) {
//		EXPECT_EQ(sample[i], raw[i]);
//	}
//}
