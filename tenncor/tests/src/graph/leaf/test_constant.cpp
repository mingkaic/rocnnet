//
// Created by Mingkai Chen on 2017-03-14.
//

#ifndef DISABLE_GRAPH_MODULE_TESTS

#include <algorithm>

#include "mocks/mock_connector.h"
#include "graph/leaf/constant.hpp"

#include "util_test.h"
#include "gtest/gtest.h"
#include "fuzz.h"


using namespace nnet;


#ifndef DISABLE_CONSTANT_TEST


// covers constant
// scalar constructor, vector constructor
TEST(CONSTANT, Constructor_D000)
{
	FUZZ::delim();
	double c = FUZZ::getDouble(1, "c")[0];
	tensorshape shape = random_def_shape();
	tensorshape part = make_partial(shape.as_list());

	size_t n = shape.n_elems();
	size_t pn = part.n_known();
	// defined shape
	std::vector<double> v = FUZZ::getDouble(n, "v");
	std::vector<double> v2 = FUZZ::getDouble(n / 2, "v2");
	std::vector<double> v3 = FUZZ::getDouble(n * 1.5, "v3");

	// partially defined shape
	std::vector<double> pv = FUZZ::getDouble(pn, "pv");
	std::vector<double> pv2 = FUZZ::getDouble(pn * 0.6, "pv2");
	std::vector<double> pv3 = FUZZ::getDouble(pn * 1.5, "pv3");

	constant<double>* res = constant<double>::get(c);
	constant<double>* res2 = constant<double>::get(v, shape);
	constant<double>* res3 = constant<double>::get(v2, shape);
	constant<double>* res4 = constant<double>::get(v3, shape);
	constant<double>* res5 = constant<double>::get(pv, part);
	constant<double>* res6 = constant<double>::get(pv2, part);
	constant<double>* res7 = constant<double>::get(pv3, part);

	EXPECT_TRUE(res->good_status()); // scalars are initialized

	std::vector<double> r2 = expose(res2);
	std::vector<double> r3 = expose(res3);
	std::vector<double> r4 = expose(res4);
	EXPECT_EQ(n, r2.size());
	EXPECT_EQ(n, r3.size());
	EXPECT_EQ(n, r4.size());
	for (size_t i = 0; i < n; i++)
	{
		EXPECT_EQ(v[i], r2[i]);
		if (i < n/2)
		{
			EXPECT_EQ(v2[i], r3[i]);
		}
		else
		{
			EXPECT_EQ((double) 0, r3[i]);
		}
		EXPECT_EQ(v3[i], r4[i]);
	}
	EXPECT_TRUE(tensorshape_equal(shape, res2->get_shape()));
	EXPECT_TRUE(tensorshape_equal(shape, res3->get_shape()));
	EXPECT_TRUE(tensorshape_equal(shape, res4->get_shape()));

	std::vector<double> r5 = expose(res5);
	std::vector<double> r6 = expose(res6);
	std::vector<double> r7 = expose(res7);

	EXPECT_EQ(pn, r5.size());
	EXPECT_EQ(pn, r6.size());
	size_t pv3n = pv3.size();
	size_t v7n = r7.size();
	ASSERT_LT(pn, pv3n);
	ASSERT_LT(pv3n, v7n);
	for (size_t i = 0; i < pn; i++)
	{
		EXPECT_EQ(pv[i], r5[i]);
		if (i < pv2.size())
		{
			EXPECT_EQ(pv2[i], r6[i]);
		}
		else
		{
			EXPECT_EQ((double) 0, r6[i]);
		}
		EXPECT_EQ(pv3[i], r7[i]);
	}
	size_t i = pn;
	for (; i < pv3n; i++)
	{
		EXPECT_EQ(pv3[i], r7[i]);
	}
	for (; i < v7n; i++)
	{
		EXPECT_EQ((double) 0, r7[i]);
	}

	// the shapes of res5 to 7 should be compatible with part
	EXPECT_TRUE(part.is_compatible_with(res5->get_shape()));
	EXPECT_TRUE(part.is_compatible_with(res3->get_shape()));
	EXPECT_TRUE(part.is_compatible_with(res4->get_shape()));

	delete res;
	delete res2;
	delete res3;
	delete res4;
	delete res5;
	delete res6;
	delete res7;
}


// covers constant
// clone and move
TEST(CONSTANT, CopyNMove_D001)
{
	FUZZ::delim();
	double c = FUZZ::getDouble(1, "c")[0];
	tensorshape shape = random_def_shape();
	tensorshape part = make_partial(shape.as_list());

	size_t n = shape.n_elems();
	size_t pn = part.n_known();
	// defined shape
	std::vector<double> v = FUZZ::getDouble(FUZZ::getInt(1, "v.size", {0.5*n, 1.5*n})[0], "v");
	// partially defined shape
	std::vector<double> pv = FUZZ::getDouble(FUZZ::getInt(1, "pv.size", {0.5*pn, 1.5*pn})[0], "pv");

	constant<double>* res = constant<double>::get(c);
	constant<double>* res2 = constant<double>::get(v, shape);
	constant<double>* res3 = constant<double>::get(pv, part);

	EXPECT_EQ(nullptr, res->clone());
	EXPECT_EQ(nullptr, res2->clone());
	EXPECT_EQ(nullptr, res3->clone());
	EXPECT_EQ(nullptr, res->move());
	EXPECT_EQ(nullptr, res2->move());
	EXPECT_EQ(nullptr, res3->move());

	delete res;
	delete res2;
	delete res3;
}


// covers constant get_gradient
TEST(CONSTANT, GetGradient_D002)
{
	FUZZ::delim();
	double c = FUZZ::getDouble(1, "c")[0];
	constant<double>* res = constant<double>::get(c);
	constant<double>* res2 = constant<double>::get(c+1);

	const tensor<double>* g1 = res->get_gradient(nullptr)->get_eval();
	const tensor<double>* g2 = res->get_gradient(res)->get_eval();
	const tensor<double>* g3 = res->get_gradient(res2)->get_eval();

	std::vector<double> gres = g1->expose();
	std::vector<double> gres1 = g2->expose();
	std::vector<double> gres2 = g3->expose();

	ASSERT_EQ((size_t) 1, gres.size());
	ASSERT_EQ((size_t) 1, gres1.size());
	ASSERT_EQ((size_t) 1, gres2.size());

	EXPECT_EQ((double) 0, gres[0]);
	EXPECT_EQ((double) 0, gres1[0]);
	EXPECT_EQ((double) 0, gres2[0]);

	delete res;
	delete res2;
}


// covers constant get_leaf
TEST(CONSTANT, GetLeaf_D003)
{
	FUZZ::delim();
	double c = FUZZ::getDouble(1, "c")[0];
	constant<double>* res = constant<double>::get(c);

	inode<double>* g1 = res->get_leaf(nullptr);

	EXPECT_TRUE(*g1 == 0.0);

	delete res;
}


// covers constant get_leaves
TEST(CONSTANT, GetLeaves_D004)
{
	FUZZ::delim();
	double c = FUZZ::getDouble(1, "c")[0];
	constant<double>* res = constant<double>::get(c);

	typename inode<double>::GRAD_CACHE leafset;
	res->get_leaves(leafset);

	EXPECT_TRUE(leafset.empty());

	delete res;
}


// covers constant commit_sudoku_sub
TEST(CONSTANT, SelfDestruct_D005)
{
	FUZZ::delim();
	double c = FUZZ::getDouble(1, "c")[0];
	constant<double>* res = constant<double>::get(c);
	constant<double>* res2 = constant<double>::get(c);
	res->is_managed_ = true;
	res2->is_managed_ = false;

	mock_connector* mconn = new mock_connector({res, res2}, "");
	delete mconn;

	EXPECT_NE(nullptr, res->get_eval());
	delete res;
}


// verifies data status
TEST(CONSTANT, Allocated_D006)
{
	FUZZ::delim();
	double c = FUZZ::getDouble(1, "c")[0];
	tensorshape shape = random_def_shape();
	tensorshape part = make_partial(shape.as_list());

	size_t n = shape.n_elems();
	size_t pn = part.n_known();
	// defined shape
	std::vector<double> v = FUZZ::getDouble(FUZZ::getInt(1, "v.size", {0.5*n, 1.5*n})[0], "v");
	// partially defined shape
	std::cout << pn << std::endl;
	std::vector<double> pv = FUZZ::getDouble(FUZZ::getInt(1, "pv.size", {0.5*pn, 1.5*pn})[0], "pv");

	constant<double>* res = constant<double>::get(c);
	constant<double>* res2 = constant<double>::get(v, shape);
	constant<double>* res3 = constant<double>::get(pv, part);

	const tensor<double>* t1 = res->get_eval();
	const tensor<double>* t2 = res2->get_eval();
	const tensor<double>* t3 = res3->get_eval();

	EXPECT_TRUE(t1->is_alloc());
	EXPECT_TRUE(t2->is_alloc());
	EXPECT_TRUE(t3->is_alloc());

	delete res;
	delete res2;
	delete res3;
}


#endif /* DISABLE_CONSTANT_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */
