//
// Created by Mingkai Chen on 2016-08-29.
//

#include <algorithm>
#include "gtest/gtest.h"
#include "graph/operation/immutable/transform.hpp"
#include "test_util.h"


TEST(OPERATION, Transpose)
{
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
	double ex1[2][4] = {
		{3, 41, 8, 18},
		{4, 6, 1, 2}
	};
	double ex2[3][4] = {
		{2, 12, 0, 0.1},
		{3, 54, 9, 29},
		{10, 11, 0, 0}
	};
	double ex3[4][5] = {
		{11, 6, 6, 12, 0},
		{22, 4, 3, 10, 2},
		{32, 45, 3, 22, 2},
		{9, 3.2, 3, 32, 1}
	};
	nnet::placeholder<double> A((std::vector<size_t>{2, 4}), "a");
	nnet::placeholder<double> B((std::vector<size_t>{3, 4}), "b");
	nnet::placeholder<double> C((std::vector<size_t>{4, 5}), "c");

	nnet::varptr<double> resa = nnet::transpose<double>(&A);
	nnet::varptr<double> resb = nnet::transpose<double>(&B);
	nnet::varptr<double> resc = nnet::transpose<double>(&C);
	A = av;
	B = bv;
	C = cv;
	nnet::tensor<double>* ta = resa->get_eval();
	nnet::tensorshape sa = ta->get_shape();
	std::vector<size_t> va = sa.as_list();
	ASSERT_EQ(va.size(), 2);
	ASSERT_EQ(va[0], 4);
	ASSERT_EQ(va[1], 2);
	for (size_t x = 0; x < 4; x++)
	{
		for (size_t y = 0; y < 2; y++)
		{
			EXPECT_EQ(ex1[y][x], ta->get({x, y}));
		}
	}
	nnet::tensor<double>* tb = resb->get_eval();
	nnet::tensorshape sb = tb->get_shape();
	std::vector<size_t> vb = sb.as_list();
	ASSERT_EQ(vb.size(), 2);
	ASSERT_EQ(vb[0], 4);
	ASSERT_EQ(vb[1], 3);
	for (size_t x = 0; x < 4; x++)
	{
		for (size_t y = 0; y < 3; y++)
		{
			EXPECT_EQ(ex2[y][x], tb->get({x, y}));
		}
	}
	nnet::tensor<double>* tc = resc->get_eval();
	nnet::tensorshape sc = tc->get_shape();
	std::vector<size_t> vc = sc.as_list();
	ASSERT_EQ(vc.size(), 2);
	ASSERT_EQ(vc[0], 5);
	ASSERT_EQ(vc[1], 4);
	for (size_t x = 0; x < 5; x++)
	{
		for (size_t y = 0; y < 4; y++)
		{
			EXPECT_EQ(ex3[y][x], tc->get({x, y}));
		}
	}
}


TEST(OPERATION, Fit)
{
	nnet::placeholder<double> A((std::vector<size_t>{1}), "a");
	nnet::placeholder<double> B((std::vector<size_t>{10, 1}), "b");
	nnet::placeholder<double> C((std::vector<size_t>{10, 5}), "c");

	nnet::placeholder<double> shape(std::vector<size_t>{10, 5}, "shape");

	nnet::varptr<double> resa = nnet::fit<double>(&A, &shape);
	nnet::varptr<double> resb = nnet::fit<double>(&B, &shape);
	nnet::varptr<double> resc = nnet::fit<double>(&C, &shape);
	shape = std::vector<double>(50, 1);
	A = std::vector<double>{1};
	B = std::vector<double>(10, 1);
	C = std::vector<double>(50, 1); // special caveat: shape must be initialized for fit to access shape TODO: fix
	std::vector<double> a = nnet::expose<double>(resa);
	nnet::tensorshape sa = resa->get_shape();
	std::vector<size_t> va = sa.as_list();
	ASSERT_EQ(va.size(), 2);
	ASSERT_EQ(va[0], 10);
	ASSERT_EQ(va[1], 5);
	size_t count = 0;
	for (double v : a)
	{
		EXPECT_EQ(1, v);
	}
	std::vector<double> b = nnet::expose<double>(resb);
	nnet::tensorshape sb = resb->get_shape();
	std::vector<size_t> vb = sb.as_list();
	ASSERT_EQ(vb.size(), 2);
	ASSERT_EQ(vb[0], 10);
	ASSERT_EQ(vb[1], 5);
	for (double v : b)
	{
		EXPECT_EQ(1, v);
	}
	std::vector<double> c = nnet::expose<double>(resc);
	nnet::tensorshape sc = resc->get_shape();
	std::vector<size_t> vc = sc.as_list();
	ASSERT_EQ(vc.size(), 2);
	ASSERT_EQ(vc[0], 10);
	ASSERT_EQ(vc[1], 5);
	for (double v : a)
	{
		EXPECT_EQ(1, v);
	}
}


TEST(OPERATION, Extend)
{
	nnet::placeholder<double> A((std::vector<size_t>{2, 1, 2}), "a");
	nnet::placeholder<double> B((std::vector<size_t>{2, 2, 1}), "b");
	nnet::placeholder<double> C((std::vector<size_t>{2, 2, 2}), "c");

	std::vector<double> t1 = {
		0.4, 0.9,
		1.2, 3.1,
	};
	std::vector<double> t2 = {
		// layer 1
		0.4, 0.9,
		1.2, 3.1,
		// layer 2
		1.9, 1.0,
		2.5, 2.0,
	};
	std::vector<double> ex1 = {
		// layer 1
		0.4, 0.9, 0.4, 0.9,
		1.2, 3.1, 1.2, 3.1,
		// layer 2
		1.9, 1.0, 1.9, 1.0,
		2.5, 2.0, 2.5, 2.0,
	};
	std::vector<double> ex2 = {
		// layer 1
		0.4, 0.9,
		0.4, 0.9,
		// layer 2
		1.2, 3.1,
		1.2, 3.1,
	};
	std::vector<double> ex3 = {
		// layer 1
		0.4, 0.9,
		1.2, 3.1,
		// layer 2
		0.4, 0.9,
		1.2, 3.1,
	};
	std::vector<double> ex4 = {
		// layer A
		// layer 1
		0.4, 0.9,
		1.2, 3.1,
		// layer 2
		1.9, 1.0,
		2.5, 2.0,
		// layer B
		// layer 1
		0.4, 0.9,
		1.2, 3.1,
		// layer 2
		1.9, 1.0,
		2.5, 2.0,
	};
	A = t1;
	B = t1;
	C = t2;

	nnet::varptr<double> e1 = nnet::extend<double>(&C, 0, 2);
	nnet::varptr<double> e2 = nnet::extend<double>(&A, 1, 2);
	nnet::varptr<double> e3 = nnet::extend<double>(&B, 2, 2);
	nnet::varptr<double> e4 = nnet::extend<double>(&C, 3, 2);

	std::vector<double> raw = nnet::expose<double>(e1);
	std::vector<size_t> ts1 = e1->get_shape().as_list();
	ASSERT_EQ(3, ts1.size());
	ASSERT_EQ(4, ts1[0]);
	for (auto it = ++ts1.begin(); it != ts1.end(); it++)
	{
		ASSERT_EQ(2, *it);
	}
	for (size_t i = 0; i < raw.size(); i++)
	{
		EXPECT_EQ(ex1[i], raw[i]);
	}

	raw = nnet::expose<double>(e2);
	std::vector<size_t> ts2 = e2->get_shape().as_list();
	ASSERT_EQ(3, ts2.size());
	for (size_t s : ts2)
	{
		ASSERT_EQ(2, s);
	}
	for (size_t i = 0; i < raw.size(); i++)
	{
		EXPECT_EQ(ex2[i], raw[i]);
	}

	raw = nnet::expose<double>(e3);
	std::vector<size_t> ts3 = e3->get_shape().as_list();
	ASSERT_EQ(3, ts3.size());
	for (size_t s : ts3)
	{
		ASSERT_EQ(2, s);
	}
	for (size_t i = 0; i < raw.size(); i++)
	{
		EXPECT_EQ(ex3[i], raw[i]);
	}

	raw = nnet::expose<double>(e4);
	std::vector<size_t> ts4 = e4->get_shape().as_list();
	ASSERT_EQ(4, ts4.size());
	for (size_t s : ts4)
	{
		ASSERT_EQ(2, s);
	}
	for (size_t i = 0; i < raw.size(); i++)
	{
		EXPECT_EQ(ex4[i], raw[i]);
	}
}


TEST(OPERATION, Compress)
{
	nnet::placeholder<double> A((std::vector<size_t>{5, 2}), "a");
	nnet::placeholder<double> B((std::vector<size_t>{2, 5}), "b");
	nnet::placeholder<double> C((std::vector<size_t>{2, 5, 2}), "c");

	std::vector<double> in1 = {
		1, 2, 3, 4, 5,
		2, 10, 23, 1, 2,
	};

	std::vector<double> exp1 = { 3, 7.6 };

	std::vector<double> in2 = {
		3, 2,
		10, 23,
		2, 1.2,
		0.5, 0.1,
		1, 2,
	};

	std::vector<double> exp2 = { 3.3, 5.66 };

	std::vector<double> in3 = {
		// layer 1
		3, 2,
		10, 23,
		2, 1.2,
		0.5, 0.1,
		1, 2,
		// layer 2
		2, 1.8,
		12, 84,
		92, 1.9,
		9, 3.14,
		70, 17.1,
	};

	std::vector<double> exp3 = {
		// layer 1
		3.3, 5.66,
		// layer 2
		37, 21.588,
	};

	A = in1;
	B = in2;
	C = in3;

	nnet::varptr<double> c1 = nnet::compress<double>(nnet::varptr<double>(&A), 0); // expect vector of 2
	nnet::varptr<double> c2 = nnet::compress<double>(nnet::varptr<double>(&B), 1); // expect vector of 2
	nnet::varptr<double> c3 = nnet::compress<double>(nnet::varptr<double>(&C), 1); // expect shape of 2, 1, 2

	std::vector<double> raw = nnet::expose<double>(c1);
	std::vector<size_t> v1 = c1->get_shape().as_list();
	ASSERT_EQ(1, v1.size());
	ASSERT_EQ(2, v1[0]);
	for (size_t i = 0; i < raw.size(); i++)
	{
		EXPECT_EQ(exp1[i], raw[i]);
	}

	raw = nnet::expose<double>(c2);
	std::vector<size_t> v2 = c2->get_shape().as_list();
	ASSERT_EQ(1, v2.size());
	ASSERT_EQ(2, v2[0]);
	for (size_t i = 0; i < raw.size(); i++)
	{
		EXPECT_EQ(exp2[i], raw[i]);
	}

	raw = nnet::expose<double>(c3);
	std::vector<size_t> v3 = c3->get_shape().as_list();
	ASSERT_EQ(3, v3.size());
	ASSERT_EQ(2, v3[0]);
	ASSERT_EQ(1, v3[1]);
	ASSERT_EQ(2, v3[2]);
	for (size_t i = 0; i < raw.size(); i++)
	{
		EXPECT_EQ(exp3[i], raw[i]);
	}
}
