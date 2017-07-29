//
// Created by Mingkai Chen on 2017-03-14.
//

#ifndef DISABLE_LEAF_MODULE_TESTS

#include <algorithm>

#include "mocks/mock_node.h"
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
	const tensor<double>* scalart = scalar.eval();
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
	const tensor<double>* scalart = sv->eval();
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
	scalart = assign1.eval();
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
	const tensor<double>* scalart = sv->eval();
	EXPECT_TRUE(scalart->is_alloc());
	EXPECT_EQ(c, scalart->expose()[0]);
	EXPECT_FALSE(nv->can_init());
	EXPECT_FALSE(nv->good_status());
	EXPECT_TRUE(civ->can_init());
	EXPECT_FALSE(civ->good_status());
	EXPECT_TRUE(rv->can_init());
	EXPECT_FALSE(rv->good_status());

	EXPECT_EQ(nullptr, scalar.eval());

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
	scalart = assign1.eval();
	EXPECT_TRUE(scalart->is_alloc());
	EXPECT_EQ(c, scalart->expose()[0]);
	EXPECT_FALSE(assign2.can_init());
	EXPECT_FALSE(assign2.good_status());
	EXPECT_TRUE(assign3.can_init());
	EXPECT_FALSE(assign3.good_status());
	EXPECT_TRUE(assign4.can_init());
	EXPECT_FALSE(assign4.good_status());

	EXPECT_EQ(nullptr, sv->eval());

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
// get_gradient
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
	mock_node exposer;

	const_init<double> cinit(c);
	rand_uniform<double> rinit(0, 1);

	variable<double> scalar(c, label1);
	variable<double> noinitv(shape, label2);
	variable<double> cinitv(shape, cinit, label3);
	variable<double> rinitv(shape, rinit, label4);

	varptr<double> wun = exposer.expose_leaf(&scalar, &scalar);
	varptr<double> wun2 = exposer.expose_leaf(&noinitv, &noinitv);
	varptr<double> wun3 = exposer.expose_leaf(&cinitv, &cinitv);
	varptr<double> wun4 = exposer.expose_leaf(&rinitv, &rinitv);

	varptr<double> zaro = exposer.expose_leaf(&scalar, nullptr);
	varptr<double> zaro2 = exposer.expose_leaf(&scalar, &noinitv);
	varptr<double> zaro3 = exposer.expose_leaf(&noinitv, nullptr);
	varptr<double> zaro4 = exposer.expose_leaf(&noinitv, &cinitv);
	varptr<double> zaro5 = exposer.expose_leaf(&cinitv, nullptr);
	varptr<double> zaro6 = exposer.expose_leaf(&cinitv, &rinitv);
	varptr<double> zaro7 = exposer.expose_leaf(&rinitv, nullptr);
	varptr<double> zaro8 = exposer.expose_leaf(&rinitv, &scalar);

	double wunvalue = expose<double>(wun)[0];
	double wunvalue2 = expose<double>(wun2)[0];
	double wunvalue3 = expose<double>(wun3)[0];
	double wunvalue4 = expose<double>(wun4)[0];
	EXPECT_TRUE(wunvalue == 1.0);
	EXPECT_TRUE(wunvalue2 == 1.0);
	EXPECT_TRUE(wunvalue3 == 1.0);
	EXPECT_TRUE(wunvalue4 == 1.0);

	double zarovalue = expose<double>(zaro)[0];
	double zarovalue2 = expose<double>(zaro2)[0];
	double zarovalue3 = expose<double>(zaro3)[0];
	double zarovalue4 = expose<double>(zaro4)[0];
	double zarovalue5 = expose<double>(zaro5)[0];
	double zarovalue6 = expose<double>(zaro6)[0];
	double zarovalue7 = expose<double>(zaro7)[0];
	double zarovalue8 = expose<double>(zaro8)[0];
	EXPECT_TRUE(zarovalue == 0.0);
	EXPECT_TRUE(zarovalue2 == 0.0);
	EXPECT_TRUE(zarovalue3 == 0.0);
	EXPECT_TRUE(zarovalue4 == 0.0);
	EXPECT_TRUE(zarovalue5 == 0.0);
	EXPECT_TRUE(zarovalue6 == 0.0);
	EXPECT_TRUE(zarovalue7 == 0.0);
	EXPECT_TRUE(zarovalue8 == 0.0);
}


// covers variable
// initialize
TEST(VARIABLE, Initialize_F004)
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
