//
// Created by Mingkai Chen on 2016-08-29.
//

#include "gtest/gtest.h"
#include "memory/session.hpp"
#include "graph/variable/variable.hpp"
#include "mock_operation.h"
using ::testing::_;


double constant = rand();
nnet::const_init<double> cinit(constant);


// Initialization Tests
TEST(VARIABLE, ConstInit)
{
	nnet::variable<double> cvar(std::vector<size_t>{3, 3}, cinit, "constant_arr");

	cvar.initialize();
	std::vector<double> raw = nnet::expose<double>(&cvar);

	ASSERT_EQ(raw.size(), 9);

	for (double elem : raw) {
		EXPECT_EQ(elem, constant);
	}
}


TEST(VARIABLE, RandomInit)
{
	nnet::random_uniform<double> rinit(0.0, 1.0);
	nnet::variable<double> rvar(std::vector<size_t>{10, 10}, rinit, "random_arr");

	rvar.initialize();
	std::vector<double> raw = nnet::expose<double>(&rvar);

	ASSERT_EQ(raw.size(), 100);

	double sum = 0;
	for (double elem : raw)
	{
		sum += elem;
		EXPECT_GT(elem, 0);
		EXPECT_LT(elem, 1);
	}
	// elem aren't all 0 (probability of this happening is virtually non-existent)
	// especially for 100 double precision floats
	// (integers have probability 2^-100 to be all zero. floats have basically 0)
	ASSERT_GT(sum/raw.size(), 0);
}


// Behavior D100
TEST(VARIABLE, InitNotify_D100)
{
	nnet::variable<double> cvar(std::vector<size_t>{5, 5}, cinit, "const_arr");
	nnet::variable<double> emptyvar(std::vector<size_t>{}, cinit, "undefined_arr");
	MockOperation* op = MockOperation::build(&cvar); // observes cvar
	MockOperation* op2 = MockOperation::build(&emptyvar); // observes cvar
	EXPECT_CALL(*op, mock_update(_, _)).Times(1);
	EXPECT_CALL(*op2, mock_update(_, _)).Times(1);
	// call update on initializations
	cvar.initialize();
	emptyvar.initialize(std::vector<size_t>{2, 3});
	delete op;
	delete op2;
}


// Behavior D200
TEST(VARIABLE, ScalarInit_D200)
{
	nnet::variable<double> scalar(10);
	ASSERT_TRUE(scalar.is_init());
}


// Prove that non-scalar constructors DON'T initialize
TEST(VARIABLE, ConstructUninit)
{
	nnet::variable<double> defvar(std::vector<size_t>{2, 2}, cinit, "init_arr");
	nnet::variable<double> pdefvar(std::vector<size_t>{0, 2}, cinit, "init_arr");
	nnet::variable<double> undefvar(std::vector<size_t>{}, cinit, "init_arr");
	ASSERT_FALSE(defvar.is_init());
	ASSERT_FALSE(pdefvar.is_init());
	ASSERT_FALSE(undefvar.is_init());
}


// Behavior D201
TEST(VARIABLE, OneVarGradient_D201)
{
	nnet::variable<double> cvar(std::vector<size_t>{3, 3}, cinit, "constant_arr");
	nnet::ivariable<double>* grad = cvar.get_gradient();
	cvar.initialize();
	std::vector<double> res = nnet::expose<double>(grad);
	ASSERT_EQ(res.size(), 1);
	EXPECT_EQ(1, res[0]);
}
