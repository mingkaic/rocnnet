//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_GRAPH_MODULE_TESTS

#include <algorithm>

#include "graph/leaf/variable.hpp"
#include "graph/connector/immutable/transform.hpp"
#include "graph/varptr.hpp"

#include "gtest/gtest.h"
#include "util_test.h"
#include "fuzz.h"


#ifndef DISABLE_TRANSFORM_TEST


using namespace nnet;


using SHAPE_CHANGE = std::function<tensorshape(tensorshape)>;
using DATA_CHANGE = std::function<std::vector<double>(std::vector<double>,tensorshape)>;

template <typename T>
using PARAM_EVAL = std::function<T(tensorshape)>;
template <typename T>
using UNARY_VAR = std::function<varptr<double>(varptr<double>,T)>;


static const double epi = std::numeric_limits<double>::epsilon();


template <typename T>
T no_param (tensorshape) { return (T)0; }


template <typename T=double>
static void unaryTransTest (std::pair<int,int> ranklimit, UNARY_VAR<T> func,
	DATA_CHANGE expect_transfer, SHAPE_CHANGE expect_shape,
	DATA_CHANGE grad_transfer, SHAPE_CHANGE grad_shape, PARAM_EVAL<T> paramer = no_param<T>)
{
	tensorshape shape = random_def_shape(ranklimit.first, ranklimit.second);
	rand_uniform<double> rinit(2, 12);
	variable<double> var(shape, rinit, "unar_var");
	var.initialize();
	tensorshape expectoshape = expect_shape(shape);
	std::vector<double> expectout = expect_transfer(expose(&var), shape);
	tensorshape gradoshape = grad_shape(var.get_gradient(&var)->get_shape());
	std::vector<double> gradout = grad_transfer(expose(var.get_gradient(&var)), shape);

	// Behavior K000
	EXPECT_EQ(nullptr, func(varptr<double>(nullptr), paramer(shape)));

	varptr<double> res = func(varptr<double>(&var), paramer(shape));
	// test forward
	tensorshape outshape = res->get_shape();
	std::vector<double> rout = expose<double>(res);
	EXPECT_TRUE(tensorshape_equal(expectoshape, outshape));
	ASSERT_EQ(expectout.size(), rout.size());
	for (size_t i = 0, n = rout.size(); i < n; i++)
	{
		EXPECT_EQ(expectout[i], rout[i]);
	}

	// test derivative
	const tensor<double>* backt = res->get_gradient(&var)->get_eval();
	tensorshape outgshape = backt->get_shape();
	std::vector<double> rgout = backt->expose();
	EXPECT_TRUE(tensorshape_equal(gradoshape, outgshape));
	ASSERT_EQ(gradout.size(), rgout.size());
	for (size_t i = 0, n = rgout.size(); i < n; i++)
	{
		EXPECT_EQ(gradout[i], rgout[i]);
	}
}


TEST(TRANSFORM, Transpose_K001)
{
	FUZZ::delim();
	DATA_CHANGE transfer =
	[](std::vector<double> in, tensorshape inshape) -> std::vector<double>
	{
		if (1 == inshape.rank())
		{
			return in;
		}
		std::vector<size_t> slist = inshape.as_list();
		tensorshape outshape({slist[1], slist[0]});
		std::vector<double> out(in.size(), 0);
		for (size_t i = 0, n = in.size(); i < n; i++)
		{
			std::vector<size_t> incoord = inshape.coordinate_from_idx(i);
			size_t j = outshape.sequential_idx({incoord[1], incoord[0]});
			out[j] = in[i];
		}
		return out;
	};
	SHAPE_CHANGE shape =
	[](tensorshape in) -> tensorshape
	{
		if (1 == in.rank())
		{
			return std::vector<size_t>{1, in.as_list()[0]};
		}
		std::vector<size_t> slist = in.as_list();
		return std::vector<size_t>{slist[1], slist[0]};
	};
	unaryTransTest<double>({1, 2},
	[](varptr<double> in,double) { return nnet::transpose(in); },
	transfer, shape, transfer, shape);
}


TEST(TRANSFORM, Fit_K002)
{
	FUZZ::delim();
	tensorshape realshape = random_def_shape(6, 6);
	rand_uniform<double> rinit(2, 12);
	variable<double> shapeholder(realshape, rinit, "shapeholder");
	shapeholder.initialize();

	PARAM_EVAL<const varptr<double> > fitparam =
	[&shapeholder](tensorshape) -> const varptr<double>
	{
		return varptr<double>(&shapeholder);
	};
	DATA_CHANGE transfer =
	[&realshape](std::vector<double> in, tensorshape inshape) -> std::vector<double>
	{
		size_t n = realshape.n_elems();
		std::vector<double> out(n, 0);
		std::vector<size_t> outlist = realshape.as_list();
		for (size_t i = 0, m = in.size(); i < m; i++)
		{
			std::vector<size_t> incoord = inshape.coordinate_from_idx(i);
			bool b = true;
			for (size_t j = 0, o = incoord.size(); j < o && b; j++)
			{
				b = incoord[j] < outlist[j];
			}
			if (b)
			{
				size_t outidx = realshape.sequential_idx(incoord);
				out[outidx] = in[i];
			}
		}
		return out;
	};
	SHAPE_CHANGE shape =
	[&realshape](tensorshape) -> tensorshape
	{
		return realshape;
	};
	unaryTransTest<const varptr<double> >({2, 5},
	[](varptr<double> in, varptr<double> watch) { return nnet::fit(in, watch); },
	transfer, shape, transfer, shape, fitparam);
}


//TEST(TRANSFORM, Extend)
//{
//	FUZZ::delim();
//	placeholder<double> A((std::vector<size_t>{2, 1, 2}), "a");
//	placeholder<double> B((std::vector<size_t>{2, 2, 1}), "b");
//	placeholder<double> C((std::vector<size_t>{2, 2, 2}), "c");
//
//	std::vector<double> t1 = {
//		0.4, 0.9,
//		1.2, 3.1,
//	};
//	std::vector<double> t2 = {
//		// layer 1
//		0.4, 0.9,
//		1.2, 3.1,
//		// layer 2
//		1.9, 1.0,
//		2.5, 2.0,
//	};
//	std::vector<double> ex1 = {
//		// layer 1
//		0.4, 0.9, 0.4, 0.9,
//		1.2, 3.1, 1.2, 3.1,
//		// layer 2
//		1.9, 1.0, 1.9, 1.0,
//		2.5, 2.0, 2.5, 2.0,
//	};
//	std::vector<double> ex2 = {
//		// layer 1
//		0.4, 0.9,
//		0.4, 0.9,
//		// layer 2
//		1.2, 3.1,
//		1.2, 3.1,
//	};
//	std::vector<double> ex3 = {
//		// layer 1
//		0.4, 0.9,
//		1.2, 3.1,
//		// layer 2
//		0.4, 0.9,
//		1.2, 3.1,
//	};
//	std::vector<double> ex4 = {
//		// layer A
//		// layer 1
//		0.4, 0.9,
//		1.2, 3.1,
//		// layer 2
//		1.9, 1.0,
//		2.5, 2.0,
//		// layer B
//		// layer 1
//		0.4, 0.9,
//		1.2, 3.1,
//		// layer 2
//		1.9, 1.0,
//		2.5, 2.0,
//	};
//	A = t1;
//	B = t1;
//	C = t2;
//
//	varptr<double> e1 = extend<double>(&C, 0, 2);
//	varptr<double> e2 = extend<double>(&A, 1, 2);
//	varptr<double> e3 = extend<double>(&B, 2, 2);
//	varptr<double> e4 = extend<double>(&C, 3, 2);
//
//	std::vector<double> raw = expose<double>(e1);
//	std::vector<size_t> ts1 = e1->get_shape().as_list();
//	ASSERT_EQ(3, ts1.size());
//	ASSERT_EQ(4, ts1[0]);
//	for (auto it = ++ts1.begin(); it != ts1.end(); it++)
//	{
//		ASSERT_EQ(2, *it);
//	}
//	for (size_t i = 0; i < raw.size(); i++)
//	{
//		EXPECT_EQ(ex1[i], raw[i]);
//	}
//
//	raw = expose<double>(e2);
//	std::vector<size_t> ts2 = e2->get_shape().as_list();
//	ASSERT_EQ(3, ts2.size());
//	for (size_t s : ts2)
//	{
//		ASSERT_EQ(2, s);
//	}
//	for (size_t i = 0; i < raw.size(); i++)
//	{
//		EXPECT_EQ(ex2[i], raw[i]);
//	}
//
//	raw = expose<double>(e3);
//	std::vector<size_t> ts3 = e3->get_shape().as_list();
//	ASSERT_EQ(3, ts3.size());
//	for (size_t s : ts3)
//	{
//		ASSERT_EQ(2, s);
//	}
//	for (size_t i = 0; i < raw.size(); i++)
//	{
//		EXPECT_EQ(ex3[i], raw[i]);
//	}
//
//	raw = expose<double>(e4);
//	std::vector<size_t> ts4 = e4->get_shape().as_list();
//	ASSERT_EQ(4, ts4.size());
//	for (size_t s : ts4)
//	{
//		ASSERT_EQ(2, s);
//	}
//	for (size_t i = 0; i < raw.size(); i++)
//	{
//		EXPECT_EQ(ex4[i], raw[i]);
//	}
//}
//
//
//TEST(TRANSFORM, Compress)
//{
//	FUZZ::delim();
//	placeholder<double> A((std::vector<size_t>{5, 2}), "a");
//	placeholder<double> B((std::vector<size_t>{2, 5}), "b");
//	placeholder<double> C((std::vector<size_t>{2, 5, 2}), "c");
//
//	std::vector<double> in1 = {
//		1, 2, 3, 4, 5,
//		2, 10, 23, 1, 2,
//	};
//
//	std::vector<double> exp1 = { 3, 7.6 };
//
//	std::vector<double> in2 = {
//		3, 2,
//		10, 23,
//		2, 1.2,
//		0.5, 0.1,
//		1, 2,
//	};
//
//	std::vector<double> exp2 = { 3.3, 5.66 };
//
//	std::vector<double> in3 = {
//		// layer 1
//		3, 2,
//		10, 23,
//		2, 1.2,
//		0.5, 0.1,
//		1, 2,
//		// layer 2
//		2, 1.8,
//		12, 84,
//		92, 1.9,
//		9, 3.14,
//		70, 17.1,
//	};
//
//	std::vector<double> exp3 = {
//		// layer 1
//		3.3, 5.66,
//		// layer 2
//		37, 21.588,
//	};
//
//	A = in1;
//	B = in2;
//	C = in3;
//
//	varptr<double> c1 = compress<double>(varptr<double>(&A), 0); // expect vector of 2
//	varptr<double> c2 = compress<double>(varptr<double>(&B), 1); // expect vector of 2
//	varptr<double> c3 = compress<double>(varptr<double>(&C), 1); // expect shape of 2, 1, 2
//
//	std::vector<double> raw = expose<double>(c1);
//	std::vector<size_t> v1 = c1->get_shape().as_list();
//	ASSERT_EQ(1, v1.size());
//	ASSERT_EQ(2, v1[0]);
//	for (size_t i = 0; i < raw.size(); i++)
//	{
//		EXPECT_EQ(exp1[i], raw[i]);
//	}
//
//	raw = expose<double>(c2);
//	std::vector<size_t> v2 = c2->get_shape().as_list();
//	ASSERT_EQ(1, v2.size());
//	ASSERT_EQ(2, v2[0]);
//	for (size_t i = 0; i < raw.size(); i++)
//	{
//		EXPECT_EQ(exp2[i], raw[i]);
//	}
//
//	raw = expose<double>(c3);
//	std::vector<size_t> v3 = c3->get_shape().as_list();
//	ASSERT_EQ(3, v3.size());
//	ASSERT_EQ(2, v3[0]);
//	ASSERT_EQ(1, v3[1]);
//	ASSERT_EQ(2, v3[2]);
//	for (size_t i = 0; i < raw.size(); i++)
//	{
//		EXPECT_EQ(exp3[i], raw[i]);
//	}
//}


#endif /* DISABLE_TRANSFORM_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */
