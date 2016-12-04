//
// Created by Mingkai Chen on 2016-08-29.
//

#include "gtest/gtest.h"
#include "graph/operation/matmul.hpp"


TEST(OPERATION, Matmul) {
	const size_t limit = 523;
	const size_t ncol = 3;
	const size_t nrow = 4;
	std::vector<double> av = {
		3, 4, 5,
		41, 6, 7,
		8, 1, 9,
		18, 2, 0
	};
	std::vector<double> bv = {
		2, 3, 10,
		12, 54, 11,
		0, 9, 0,
		0.1, 29, 0
	};
	std::vector<double> cv = {
		11, 22, 32, 9,
		6, 4, 45, 3.2,
		6, 3, 3, 3
	};
	double ex1[4][4] = {
		{68, 307, 36, 116.3},
		{170, 893, 54, 178.1},
		{109, 249, 9, 29.8},
		{42, 324, 18, 59.8}
	};
	double ex2[3][3] = {
		{499.8, 2817, 481},
		{80.2, 403, 106},
		{94, 474, 127}
	};
	double ex3[3][3] = {
		{1353, 226, 497},
		{599.6, 99.4, 463},
		{219, 51, 78}
	};
	const size_t supersize = ncol * nrow;
	nnet::placeholder<double> A((std::vector<size_t>{ncol, nrow}), "a");
	nnet::placeholder<double> B((std::vector<size_t>{ncol, nrow}), "b");
	nnet::placeholder<double> C((std::vector<size_t>{nrow, ncol}), "c");
	nnet::varptr<double> ans1 = nnet::matmul<double>::build(&A, &B, false, true); // output is 4x4
	nnet::varptr<double> ans2 = nnet::matmul<double>::build(&A, &B, true); // output is 3x3
	nnet::varptr<double> ans3 = nnet::matmul<double>::build(&C, &A); // output is 3x3

	// didn't initialize
	EXPECT_DEATH({ nnet::expose<double>(ans1); }, ".*");
	EXPECT_DEATH({ nnet::expose<double>(ans2); }, ".*");
	EXPECT_DEATH({ nnet::expose<double>(ans3); }, ".*");
	A = av;
	B = bv;
	C = cv;

	// evaluates
	nnet::tensor<double>* t1 = ans1->get_eval();
	nnet::tensorshape s1 = t1->get_shape();
	std::vector<size_t> v1 = s1.as_list();
	ASSERT_EQ(v1.size(), 2);
	ASSERT_EQ(v1[0], nrow);
	ASSERT_EQ(v1[1], nrow);

	nnet::tensor<double>* t2 = ans2->get_eval();
	nnet::tensorshape s2 = t2->get_shape();
	std::vector<size_t> v2 = s2.as_list();
	ASSERT_EQ(v2.size(), 2);
	ASSERT_EQ(v2[0], ncol);
	ASSERT_EQ(v2[1], ncol);

	nnet::tensor<double>* t3 = ans3->get_eval();
	nnet::tensorshape s3 = t3->get_shape();
	std::vector<size_t> v3 = s3.as_list();
	ASSERT_EQ(v3.size(), 2);
	ASSERT_EQ(v3[0], ncol);
	ASSERT_EQ(v3[1], ncol);

	for (size_t x = 0; x < nrow; x++) {
		for (size_t y = 0; y < nrow; y++) {
			EXPECT_EQ(ex1[y][x], t1->get({x, y}));
		}
	}
	for (size_t x = 0; x < ncol; x++) {
		for (size_t y = 0; y < ncol; y++) {
			EXPECT_EQ(ex2[y][x], t2->get({x, y}));
			EXPECT_EQ(ex3[y][x], t3->get({x, y}));
		}
	}
}


TEST(OPERATION, Matmul2) {
	const size_t limit = 523;
	std::vector<double> av = {
		3, 4,
		41, 6,
		8, 1,
		18, 2,
	};
	std::vector<double> bv = {
		2, 3, 10,
		12, 54, 11,
		0, 9, 0,
		0.1, 29, 0
	};
	std::vector<double> cv = {
		11, 22, 32, 9,
		6, 4, 45, 3.2,
		6, 3, 3, 3,
		12, 10, 22, 32,
		0, 2, 2, 1
	};
	double ex1[2][3] = {
		{499.8, 2817, 481},
		{80.2, 403, 106}
	};
	double ex2[3][5] = {
		{286.9, 60.32, 48.3, 147.2, 24.1},
		{1770, 731.8, 294, 1702, 155},
		{352, 104, 93, 230, 22}
	};
	double ex3[5][2] = {
		{1353, 226},
		{599.6, 99.4},
		{219, 51},
		{1198, 194},
		{116, 16}
	};
	nnet::placeholder<double> A((std::vector<size_t>{2, 4}), "a");
	nnet::placeholder<double> B((std::vector<size_t>{3, 4}), "b");
	nnet::placeholder<double> C((std::vector<size_t>{4, 5}), "c");
	nnet::varptr<double> ans1 = nnet::matmul<double>::build(&A, &B, true); // output is 2x3 (row by col)
	nnet::varptr<double> ans2 = nnet::matmul<double>::build(&B, &C, true, true); // output is 3x5
	nnet::varptr<double> ans3 = nnet::matmul<double>::build(&C, &A); // output is 5x2

	// didn't initialize
	EXPECT_DEATH({ nnet::expose<double>(ans1); }, ".*");
	EXPECT_DEATH({ nnet::expose<double>(ans2); }, ".*");
	EXPECT_DEATH({ nnet::expose<double>(ans3); }, ".*");
	A = av;
	B = bv;
	C = cv;

	// evaluates
	nnet::tensor<double>* t1 = ans1->get_eval();
	nnet::tensorshape s1 = t1->get_shape();
	std::vector<size_t> v1 = s1.as_list();
	ASSERT_EQ(v1.size(), 2);
	ASSERT_EQ(v1[0], 3);
	ASSERT_EQ(v1[1], 2);
	for (size_t x = 0; x < 3; x++) {
		for (size_t y = 0; y < 2; y++) {
			EXPECT_EQ(ex1[y][x], t1->get({x, y}));
		}
	}

	nnet::tensor<double>* t2 = ans2->get_eval();
	nnet::tensorshape s2 = t2->get_shape();
	std::vector<size_t> v2 = s2.as_list();
	ASSERT_EQ(v2.size(), 2);
	ASSERT_EQ(v2[0], 5);
	ASSERT_EQ(v2[1], 3);
	std::stringstream res;
	for (size_t x = 0; x < 5; x++) {
		for (size_t y = 0; y < 3; y++) {
			EXPECT_EQ(ex2[y][x], t2->get({x, y}));
		}
	}

	nnet::tensor<double>* t3 = ans3->get_eval();
	nnet::tensorshape s3 = t3->get_shape();
	std::vector<size_t> v3 = s3.as_list();
	ASSERT_EQ(v3.size(), 2);
	ASSERT_EQ(v3[0], 2);
	ASSERT_EQ(v3[1], 5);
	for (size_t y = 0; y < 5; y++) {
		for (size_t x = 0; x < 2; x++) {
			EXPECT_EQ(ex3[y][x], t3->get({x, y}));
		}
	}
}