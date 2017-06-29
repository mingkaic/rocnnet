//
// Created by Mingkai Chen on 2017-03-14.
//

#ifndef DISABLE_LEAF_MODULE_TESTS

#include <algorithm>

#include "mocks/mock_connector.h"
#include "graph/leaf/variable.hpp"

#include "util_test.h"
#include "gtest/gtest.h"
#include "fuzz.h"


using namespace nnet;


#ifndef DISABLE_VARIABLE_TEST


// covers variable
// scalar, no init, and init constructors
TEST(VARIABLE, Constructor_F000)
{
	FUZZ::reset_logger();
	std::vector<size_t> strns = FUZZ::getInt(4, "strns", {14, 29});
	std::string label1 = FUZZ::getString(strns[0], "label1");
	std::string label2 = FUZZ::getString(strns[1], "label2");
	std::string label3 = FUZZ::getString(strns[2], "label3");
	std::string label4 = FUZZ::getString(strns[3], "label4");
	tensorshape shape = random_def_shape();
	double c = FUZZ::getDouble(1, "c")[0];

	const_init<double> cinit(c);
	rand_uniform<double> rinit(0, 1);

	variable<double> scalar(c, label1);
	variable<double> noinitv(shape, label2);
	variable<double> cinitv(shape, cinit, label3);
	variable<double> rinitv(shape, rinit, label4);

	EXPECT_TRUE(scalar.can_init());
	EXPECT_TRUE(scalar.good_status());
	const tensor<double>* scalart = scalar.get_eval();
	EXPECT_TRUE(scalart->is_alloc());
	EXPECT_EQ(c, scalart->expose()[0]);

	EXPECT_FALSE(noinitv.can_init());
	EXPECT_FALSE(noinitv.good_status());

	EXPECT_TRUE(cinitv.can_init());
	EXPECT_FALSE(cinitv.good_status());

	EXPECT_TRUE(rinitv.can_init());
	EXPECT_FALSE(rinitv.good_status());
}


// covers clone function
TEST(VARIABLE, Copy_F001)
{
	FUZZ::reset_logger();
	std::vector<size_t> strns = FUZZ::getInt(4, "strns", {14, 29});
	std::string label1 = FUZZ::getString(strns[0], "label1");
	std::string label2 = FUZZ::getString(strns[1], "label2");
	std::string label3 = FUZZ::getString(strns[2], "label3");
	std::string label4 = FUZZ::getString(strns[3], "label4");
	tensorshape shape = random_def_shape();
	double c = FUZZ::getDouble(1, "c")[0];

	const_init<double> cinit(c);
	rand_uniform<double> rinit(0, 1);

	variable<double> assign1(0);
	variable<double> assign2(0);
	variable<double> assign3(0);
	variable<double> assign4(0);

	variable<double> scalar(c, label1);
	variable<double> noinitv(shape, label2);
	variable<double> cinitv(shape, cinit, label3);
	variable<double> rinitv(shape, rinit, label4);

	variable<double>* sv = scalar.clone();
	variable<double>* nv = noinitv.clone();
	variable<double>* civ = cinitv.clone();
	variable<double>* rv = rinitv.clone();

	assign1 = scalar;
	assign2 = noinitv;
	assign3 = cinitv;
	assign4 = rinitv;

	EXPECT_TRUE(sv->can_init());
	EXPECT_TRUE(sv->good_status());
	const tensor<double>* scalart = sv->get_eval();
	EXPECT_TRUE(scalart->is_alloc());
	EXPECT_EQ(c, scalart->expose()[0]);
	EXPECT_FALSE(nv->can_init());
	EXPECT_FALSE(nv->good_status());
	EXPECT_TRUE(civ->can_init());
	EXPECT_FALSE(civ->good_status());
	EXPECT_TRUE(rv->can_init());
	EXPECT_FALSE(rv->good_status());

	EXPECT_TRUE(assign1.can_init());
	EXPECT_TRUE(assign1.good_status());
	scalart = assign1.get_eval();
	EXPECT_TRUE(scalart->is_alloc());
	EXPECT_EQ(c, scalart->expose()[0]);
	EXPECT_FALSE(assign2.can_init());
	EXPECT_FALSE(assign2.good_status());
	EXPECT_TRUE(assign3.can_init());
	EXPECT_FALSE(assign3.good_status());
	EXPECT_TRUE(assign4.can_init());
	EXPECT_FALSE(assign4.good_status());

	delete sv;
	delete nv;
	delete civ;
	delete rv;
}


// covers move function
TEST(VARIABLE, Move_F001)
{
	FUZZ::reset_logger();
	std::vector<size_t> strns = FUZZ::getInt(4, "strns", {14, 29});
	std::string label1 = FUZZ::getString(strns[0], "label1");
	std::string label2 = FUZZ::getString(strns[1], "label2");
	std::string label3 = FUZZ::getString(strns[2], "label3");
	std::string label4 = FUZZ::getString(strns[3], "label4");
	tensorshape shape = random_def_shape();
	double c = FUZZ::getDouble(1, "c")[0];

	const_init<double> cinit(c);
	rand_uniform<double> rinit(0, 1);

	variable<double> assign1(0);
	variable<double> assign2(0);
	variable<double> assign3(0);
	variable<double> assign4(0);

	variable<double> scalar(c, label1);
	variable<double> noinitv(shape, label2);
	variable<double> cinitv(shape, cinit, label3);
	variable<double> rinitv(shape, rinit, label4);

	variable<double>* sv = scalar.move();
	variable<double>* nv = noinitv.move();
	variable<double>* civ = cinitv.move();
	variable<double>* rv = rinitv.move();

	EXPECT_FALSE(scalar.can_init());
	EXPECT_TRUE(scalar.good_status());
	EXPECT_FALSE(noinitv.can_init());
	EXPECT_FALSE(noinitv.good_status());
	EXPECT_FALSE(cinitv.can_init());
	EXPECT_FALSE(cinitv.good_status());
	EXPECT_FALSE(rinitv.can_init());
	EXPECT_FALSE(rinitv.good_status());

	EXPECT_TRUE(sv->can_init());
	EXPECT_TRUE(sv->good_status());
	const tensor<double>* scalart = sv->get_eval();
	EXPECT_TRUE(scalart->is_alloc());
	EXPECT_EQ(c, scalart->expose()[0]);
	EXPECT_FALSE(nv->can_init());
	EXPECT_FALSE(nv->good_status());
	EXPECT_TRUE(civ->can_init());
	EXPECT_FALSE(civ->good_status());
	EXPECT_TRUE(rv->can_init());
	EXPECT_FALSE(rv->good_status());

	EXPECT_EQ(nullptr, scalar.get_eval());

	assign1 = std::move(*sv);
	assign2 = std::move(*nv);
	assign3 = std::move(*civ);
	assign4 = std::move(*rv);

	EXPECT_FALSE(sv->can_init());
	EXPECT_TRUE(sv->good_status());
	EXPECT_FALSE(nv->can_init());
	EXPECT_FALSE(nv->good_status());
	EXPECT_FALSE(civ->can_init());
	EXPECT_FALSE(civ->good_status());
	EXPECT_FALSE(rv->can_init());
	EXPECT_FALSE(rv->good_status());

	EXPECT_TRUE(assign1.can_init());
	EXPECT_TRUE(assign1.good_status());
	scalart = assign1.get_eval();
	EXPECT_TRUE(scalart->is_alloc());
	EXPECT_EQ(c, scalart->expose()[0]);
	EXPECT_FALSE(assign2.can_init());
	EXPECT_FALSE(assign2.good_status());
	EXPECT_TRUE(assign3.can_init());
	EXPECT_FALSE(assign3.good_status());
	EXPECT_TRUE(assign4.can_init());
	EXPECT_FALSE(assign4.good_status());

	EXPECT_EQ(nullptr, sv->get_eval());

	delete sv;
	delete nv;
	delete civ;
	delete rv;
}


// covers variable
// set_initializer, initialize
TEST(VARIABLE, SetInit_F002)
{
	FUZZ::reset_logger();
	std::vector<size_t> strns = FUZZ::getInt(4, "strns", {14, 29});
	std::string label1 = FUZZ::getString(strns[0], "label1");
	std::string label2 = FUZZ::getString(strns[1], "label2");
	std::string label3 = FUZZ::getString(strns[2], "label3");
	std::string label4 = FUZZ::getString(strns[3], "label4");
	tensorshape shape = random_def_shape();
	double c = FUZZ::getDouble(1, "c")[0];

	const_init<double> cinit(c);
	rand_uniform<double> rinit(0, 1);

	variable<double> scalar(c, label1);
	variable<double> noinitv(shape, label2);
	variable<double> cinitv(shape, cinit, label3);
	variable<double> rinitv(shape, rinit, label4);

	scalar.set_initializer(rinit);
	noinitv.set_initializer(cinit);
	cinitv.set_initializer(rinit);
	rinitv.set_initializer(cinit);

	scalar.initialize();
	noinitv.initialize();
	cinitv.initialize();
	rinitv.initialize();

	std::vector<double> rv = expose(&scalar);
	std::vector<double> cv = expose(&noinitv);
	std::vector<double> rv2 = expose(&cinitv);
	std::vector<double> cv2 = expose(&rinitv);

	EXPECT_EQ((size_t) 1, rv.size());
	EXPECT_TRUE(tensorshape_equal(shape, noinitv.get_shape()));
	EXPECT_TRUE(tensorshape_equal(shape, cinitv.get_shape()));
	EXPECT_TRUE(tensorshape_equal(shape, rinitv.get_shape()));

	EXPECT_NE(c, rv[0]);
	for (size_t i = 0, n = shape.n_elems(); i < n; i++)
	{
		EXPECT_EQ(c, cv[i]);
	}
	for (size_t i = 0, n = shape.n_elems(); i < n; i++)
	{
		EXPECT_NE(c, rv2[i]);
	}
	for (size_t i = 0, n = shape.n_elems(); i < n; i++)
	{
		EXPECT_EQ(c, cv2[i]);
	}
}


// covers variable
// get_leaf
TEST(VARIABLE, GetLeaf_F003)
{
	FUZZ::reset_logger();
	std::vector<size_t> strns = FUZZ::getInt(4, "strns", {14, 29});
	std::string label1 = FUZZ::getString(strns[0], "label1");
	std::string label2 = FUZZ::getString(strns[1], "label2");
	std::string label3 = FUZZ::getString(strns[2], "label3");
	std::string label4 = FUZZ::getString(strns[3], "label3");
	tensorshape shape = random_def_shape();
	double c = FUZZ::getDouble(1, "c")[0];

	const_init<double> cinit(c);
	rand_uniform<double> rinit(0, 1);

	variable<double> scalar(c, label1);
	variable<double> noinitv(shape, label2);
	variable<double> cinitv(shape, cinit, label3);
	variable<double> rinitv(shape, rinit, label4);

	varptr<double> wun;
	varptr<double> wun2;
	varptr<double> wun3;
	varptr<double> wun4;
	scalar.get_leaf(wun, &scalar);
	noinitv.get_leaf(wun2, &noinitv);
	cinitv.get_leaf(wun3, &cinitv);
	rinitv.get_leaf(wun4, &rinitv);

	varptr<double> zaro;
	varptr<double> zaro2;
	varptr<double> zaro3;
	varptr<double> zaro4;
	varptr<double> zaro5;
	varptr<double> zaro6;
	varptr<double> zaro7;
	varptr<double> zaro8;
	scalar.get_leaf(zaro, nullptr);
	scalar.get_leaf(zaro2, &noinitv);
	noinitv.get_leaf(zaro3, nullptr);
	noinitv.get_leaf(zaro4, &cinitv);
	cinitv.get_leaf(zaro5, nullptr);
	cinitv.get_leaf(zaro6, &rinitv);
	rinitv.get_leaf(zaro7, nullptr);
	rinitv.get_leaf(zaro8, &scalar);

	EXPECT_TRUE(*wun == 1.0);
	EXPECT_TRUE(*wun2 == 1.0);
	EXPECT_TRUE(*wun3 == 1.0);
	EXPECT_TRUE(*wun4 == 1.0);

	EXPECT_TRUE(*zaro == 0.0);
	EXPECT_TRUE(*zaro2 == 0.0);
	EXPECT_TRUE(*zaro3 == 0.0);
	EXPECT_TRUE(*zaro4 == 0.0);
	EXPECT_TRUE(*zaro5 == 0.0);
	EXPECT_TRUE(*zaro6 == 0.0);
	EXPECT_TRUE(*zaro7 == 0.0);
	EXPECT_TRUE(*zaro8 == 0.0);
}


// covers variable
// get_leaves
TEST(VARIABLE, GetLeaves_F004)
{
	FUZZ::reset_logger();
	std::vector<size_t> strns = FUZZ::getInt(4, "strns", {14, 29});
	std::string label1 = FUZZ::getString(strns[0]);
	std::string label2 = FUZZ::getString(strns[1]);
	std::string label3 = FUZZ::getString(strns[2]);
	std::string label4 = FUZZ::getString(strns[3]);
	tensorshape shape = random_def_shape();
	double c = FUZZ::getDouble(1, "c")[0];

	const_init<double> cinit(c);
	rand_uniform<double> rinit(0, 1);

	variable<double> scalar(c, label1);
	variable<double> noinitv(shape, label2);
	variable<double> cinitv(shape, cinit, label3);
	variable<double> rinitv(shape, rinit, label4);

	typename inode<double>::GRAD_CACHE leafset;
	scalar.get_leaves(leafset);
	EXPECT_TRUE(leafset.end() != leafset.find(&scalar));
	noinitv.get_leaves(leafset);
	EXPECT_TRUE(leafset.end() != leafset.find(&noinitv));
	cinitv.get_leaves(leafset);
	EXPECT_TRUE(leafset.end() != leafset.find(&cinitv));
	rinitv.get_leaves(leafset);
	EXPECT_TRUE(leafset.end() != leafset.find(&rinitv));
}


// covers variable
// initialize
TEST(VARIABLE, Initialize_F005)
{
	mocker::usage_.clear();
	FUZZ::reset_logger();
	std::vector<size_t> strns = FUZZ::getInt(5, "strns", {14, 29});
	std::string label1 = FUZZ::getString(strns[0]);
	std::string label2 = FUZZ::getString(strns[1]);
	std::string label3 = FUZZ::getString(strns[2]);
	std::string label4 = FUZZ::getString(strns[3]);
	std::string label5 = FUZZ::getString(strns[4]);
	tensorshape shape = random_def_shape();
	tensorshape shape2 = random_def_shape();
	tensorshape part = make_partial(shape2.as_list());
	double c = FUZZ::getDouble(1, "c")[0];

	const_init<double> cinit(c);
	rand_uniform<double> rinit(0, 1);

	variable<double> scalar(c, label1);
	variable<double> noinitv(shape, label2);
	variable<double> cinitv(part, cinit, label3);
	variable<double> rinitv(shape, rinit, label4);

	mock_connector conn({&scalar, &noinitv, &cinitv, &rinitv}, label5);
	conn.inst_ = "conn";
	scalar.initialize();
	EXPECT_TRUE(mocker::EXPECT_CALL("conn::update1", 1));
//	EXPECT_DEATH(noinitv.initialize(), ".*";
	noinitv.set_initializer(cinit);
	noinitv.initialize();
	EXPECT_TRUE(mocker::EXPECT_CALL("conn::update1", 2));
	EXPECT_THROW(cinitv.initialize(), std::exception);
	cinitv.initialize(shape2);
	EXPECT_TRUE(mocker::EXPECT_CALL("conn::update1", 3));
	EXPECT_THROW(rinitv.initialize(part), std::exception);
	rinitv.initialize();
	EXPECT_TRUE(mocker::EXPECT_CALL("conn::update1", 4));

	std::vector<double> sv = expose(&scalar);
	std::vector<double> nv = expose(&noinitv);
	std::vector<double> cv = expose(&cinitv);
	std::vector<double> rv = expose(&rinitv);

	EXPECT_EQ((size_t) 1, sv.size());
	EXPECT_TRUE(tensorshape_equal(shape, noinitv.get_shape()));
	EXPECT_TRUE(tensorshape_equal(shape2, cinitv.get_shape()));
	EXPECT_TRUE(tensorshape_equal(shape, rinitv.get_shape()));
	EXPECT_EQ(c, sv[0]);
	for (size_t i = 0, n = shape.n_elems(); i < n; i++)
	{
		EXPECT_EQ(c, nv[i]);
	}
	for (size_t i = 0, n = shape2.n_elems(); i < n; i++)
	{
		EXPECT_EQ(c, cv[i]);
	}
	for (size_t i = 0, n = shape.n_elems(); i < n; i++)
	{
		EXPECT_NE(c, rv[i]);
		EXPECT_LE(0, rv[i]);
		EXPECT_GE(1, rv[i]);
	}
}


#endif /* DISABLE_VARIABLE_TEST */


#endif /* DISABLE_LEAF_MODULE_TESTS */
