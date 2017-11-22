//
// Created by Mingkai Chen on 2017-03-14.
//

#ifndef DISABLE_LEAF_MODULE_TESTS

#include <algorithm>

#include "mocks/mock_ivariable.h"
#include "graph/varptr.hpp"

#include "gtest/gtest.h"
#include "fuzz.h"


#ifndef DISABLE_IVARIABLE_TEST


TEST(IVARIABLE, Copy_E000)
{
	FUZZ::reset_logger();
	std::string label1 = FUZZ::getString(FUZZ::getInt(1, "label1.size", {14, 29})[0], "label1");
	tensorshape shape = random_def_shape();
	double c = FUZZ::getDouble(1, "c")[0];

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

	tensor<double> ct(std::vector<size_t>{1});
	tensor<double> ct2(std::vector<size_t>{1});
	initializer<double>* ci = cpy2->get_initializer();
	initializer<double>* ai = assign2.get_initializer();
	tensor<double>* ctptr = &ct;
	tensor<double>* ct2ptr = &ct2;
	(*ci)(*ctptr);
	(*ai)(*ct2ptr);
	EXPECT_EQ(c, ct.expose()[0]);
	EXPECT_EQ(c, ct2.expose()[0]);

	delete cpy;
	delete cpy2;
}


TEST(IVARIABLE, Move_E000)
{
	FUZZ::reset_logger();
	std::string label1 = FUZZ::getString(FUZZ::getInt(1, "label1.size", {14, 29})[0], "label1");
	tensorshape shape = random_def_shape();
	double c = FUZZ::getDouble(1, "c")[0];

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
	tensor<double> ct(std::vector<size_t>{1});
	initializer<double>* mi = mv2->get_initializer();
	tensor<double>* ctptr = &ct;
	(*mi)(*ctptr);
	EXPECT_EQ(c, ct.expose()[0]);
	EXPECT_EQ(ii, mi);
	EXPECT_EQ(nullptr, inited.get_initializer());
	EXPECT_EQ(nullptr, inited.eval());

	assign = std::move(*mv);
	assign2 = std::move(*mv2);

	EXPECT_EQ(noi, assign.get_initializer());
	tensor<double> ct2(std::vector<size_t>{1});
	initializer<double>* ai = assign2.get_initializer();
	tensor<double>* ct2ptr = &ct2;
	(*ai)(*ct2ptr);
	EXPECT_EQ(c, ct2.expose()[0]);
	EXPECT_EQ(ii, ai);
	EXPECT_EQ(nullptr, mv2->get_initializer());
	EXPECT_EQ(nullptr, mv2->eval());

	delete mv;
	delete mv2;
}


TEST(IVARIABLE, Initialize_E001)
{
	FUZZ::reset_logger();
	std::string label1 = FUZZ::getString(FUZZ::getInt(1, "label1.size", {14, 29})[0], "label1");
	tensorshape shape = random_def_shape();
	double c = FUZZ::getDouble(1, "c")[0];
	const_init<double>* cinit = new const_init<double>(c);

	mock_ivariable noinit(shape, nullptr, label1);
	mock_ivariable inited(shape, cinit, label1);

	EXPECT_FALSE(noinit.can_init());
	EXPECT_TRUE(inited.can_init());
}


TEST(IVARIABLE, GetGradient_E002)
{
	FUZZ::reset_logger();
	std::string label1 = FUZZ::getString(FUZZ::getInt(1, "label1.size", {14, 29})[0], "label1");
	tensorshape shape = random_def_shape();
	double c = FUZZ::getDouble(1, "c")[0];
	const_init<double>* cinit = new const_init<double>(c);

	mock_ivariable noinit(shape, nullptr, label1);
	mock_ivariable inited(shape, cinit, label1);

	const tensor<double>* wun = noinit.derive(&noinit)->eval();
	const tensor<double>* wuntoo = inited.derive(&inited)->eval();
	const tensor<double>* zaro = noinit.derive(&inited)->eval();
	const tensor<double>* zarotoo = inited.derive(&noinit)->eval();

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


#endif /* DISABLE_LEAF_MODULE_TESTS */
