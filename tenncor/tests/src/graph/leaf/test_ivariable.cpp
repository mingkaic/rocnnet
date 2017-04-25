//
// Created by Mingkai Chen on 2017-03-14.
//

#ifndef DISABLE_GRAPH_MODULE_TESTS

#include <algorithm>

#include "mocks/mock_ivariable.h"

#include "gtest/gtest.h"
#include "fuzz.h"


//#define DISABLE_IVARIABLE_TEST
#ifndef DISABLE_IVARIABLE_TEST


TEST(IVARIABLE, Copy_E000)
{
	FUZZ::delim();
	std::string label1 = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
	tensorshape shape = random_def_shape();
	double c = FUZZ::getDouble(1)[0];

	const_init<double>* cinit = new const_init<double>(c);

	mock_ivariable assign(shape, nullptr, "");
	mock_ivariable assign2(shape, nullptr, "");
	mock_ivariable noinit(shape, nullptr, label1);
	mock_ivariable inited(shape, cinit, label1);

	mock_ivariable* cpy = static_cast<mock_ivariable*>(noinit.clone());
	mock_ivariable* cpy2 = static_cast<mock_ivariable*>(inited.clone());
	assign = noinit;
	assign2 = inited;

	initializer<double>* noi = noinit.get_initializer();
	EXPECT_EQ(noi, cpy->get_initializer());
	EXPECT_EQ(noi, assign.get_initializer());
	EXPECT_EQ(nullptr, noi);

	tensor<double> ct({1});
	tensor<double> ct2({1});
	initializer<double>* ci = cpy2->get_initializer();
	initializer<double>* ai = assign2.get_initializer();
	(*ci)(ct);
	(*ai)(ct2);
	EXPECT_EQ(c, ct.expose()[0]);
	EXPECT_EQ(c, ct2.expose()[0]);

	delete cpy;
	delete cpy2;
}


TEST(IVARIABLE, Move_E000)
{
	FUZZ::delim();
	std::string label1 = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
	tensorshape shape = random_def_shape();
	double c = FUZZ::getDouble(1)[0];

	const_init<double>* cinit = new const_init<double>(c);

	mock_ivariable assign(shape, nullptr, "");
	mock_ivariable assign2(shape, nullptr, "");
	mock_ivariable noinit(shape, nullptr, label1);
	mock_ivariable inited(shape, cinit, label1);

	initializer<double>* noi = noinit.get_initializer();
	initializer<double>* ii = inited.get_initializer();

	EXPECT_EQ(nullptr, noi);

	mock_ivariable* mv = static_cast<mock_ivariable*>(noinit.move());
	mock_ivariable* mv2 = static_cast<mock_ivariable*>(inited.move());

	EXPECT_EQ(noi, mv->get_initializer());
	tensor<double> ct({1});
	initializer<double>* mi = mv2->get_initializer();
	(*mi)(ct);
	EXPECT_EQ(c, ct.expose()[0]);
	EXPECT_EQ(ii, mi);
	EXPECT_EQ(nullptr, inited.get_initializer());
	EXPECT_EQ(nullptr, inited.get_eval());

	assign = std::move(*mv);
	assign2 = std::move(*mv2);

	EXPECT_EQ(noi, assign.get_initializer());
	tensor<double> ct2({1});
	initializer<double>* ai = assign2.get_initializer();
	(*ai)(ct2);
	EXPECT_EQ(c, ct2.expose()[0]);
	EXPECT_EQ(ii, ai);
	EXPECT_EQ(nullptr, mv2->get_initializer());
	EXPECT_EQ(nullptr, mv2->get_eval());

	delete mv;
	delete mv2;
}


TEST(IVARIABLE, Initialize_E001)
{
	FUZZ::delim();
	std::string label1 = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
	tensorshape shape = random_def_shape();
	double c = FUZZ::getDouble(1)[0];
	const_init<double>* cinit = new const_init<double>(c);

	mock_ivariable noinit(shape, nullptr, label1);
	mock_ivariable inited(shape, cinit, label1);

	EXPECT_FALSE(noinit.can_init());
	EXPECT_TRUE(inited.can_init());
}


TEST(IVARIABLE, GetGradient_E002)
{
	FUZZ::delim();
	std::string label1 = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
	tensorshape shape = random_def_shape();
	double c = FUZZ::getDouble(1)[0];
	const_init<double>* cinit = new const_init<double>(c);

	mock_ivariable noinit(shape, nullptr, label1);
	mock_ivariable inited(shape, cinit, label1);

	const tensor<double>* wun = noinit.get_gradient(&noinit)->get_eval();
	const tensor<double>* wuntoo = inited.get_gradient(&inited)->get_eval();
	const tensor<double>* zaro = noinit.get_gradient(&inited)->get_eval();
	const tensor<double>* zarotoo = inited.get_gradient(&noinit)->get_eval();

	EXPECT_EQ((size_t) 1, wun->n_elems());
	EXPECT_EQ((size_t) 1, wuntoo->n_elems());
	EXPECT_EQ((size_t) 1, zaro->n_elems());
	EXPECT_EQ((size_t) 1, zarotoo->n_elems());
	EXPECT_EQ(1.0, wun->expose()[0]);
	EXPECT_EQ(1.0, wuntoo->expose()[0]);
	EXPECT_EQ(0.0, zaro->expose()[0]);
	EXPECT_EQ(0.0, zarotoo->expose()[0]);
}


#endif /* DISABLE_IVARIABLE_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */
