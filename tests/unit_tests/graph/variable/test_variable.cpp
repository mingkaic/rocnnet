//
// Created by Mingkai Chen on 2016-08-29.
//

#include "gtest/gtest.h"
#include "memory/session.hpp"
#include "graph/variable/variable.hpp"
#include "executor/varptr.hpp"


//TEST(VARIABLE, GetIndex) {
//	nnet::session& sess = nnet::session::get_instance();
//	const double constant = (double) rand();
//	nnet::const_init<double> init(constant);
//	size_t nrows = 3;
//	size_t ncols = 4;
//	size_t ndeps = 5; // depth
//
//	nnet::variable<double>* cvar =
//			new nnet::variable<double>(std::vector<size_t>{nrows, ncols, ndeps}, init, "constant_arr");
//
//	sess.initialize_all<double>();
//	nnet::tensor<double>* raw = cvar->get_eval();
//
//	ASSERT_EQ(raw->n_elems(), nrows*ncols*ndeps);
//
//	for (size_t i = 0; i < nrows; i++) {
//		for (size_t j = 0; j < ncols; j++) {
//			for (size_t k = 0; k < ndeps; k++) {
//				std::cout << i << " " << j << " " << k << "\n";
//				EXPECT_EQ(constant, raw->get({i,j,k}));
//			}
//		}
//	}
//	delete cvar;
//}

//
//TEST(VARIABLE, ConstInit) {
//	nnet::session& sess = nnet::session::get_instance();
//	const double constant = (double) rand();
//	nnet::const_init<double> init(constant);
//
//	nnet::variable<double>* cvar = new nnet::variable<double>(std::vector<size_t>{3, 3}, init, "constant_arr");
//
//	sess.initialize_all<double>();
//	std::vector<double> raw = nnet::expose<double>(cvar);
//
//	ASSERT_EQ(raw.size(), 9);
//
//	for (double elem : raw) {
//		EXPECT_EQ(elem, constant);
//	}
//	delete cvar;
//}
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
