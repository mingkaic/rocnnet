//
// Created by Mingkai Chen on 2017-09-12.
//

#ifndef DISABLE_CONNECTOR_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"
#include "util_test.h"
#include "fuzz.h"

#include "graph/leaf/constant.hpp"
#include "graph/connector/immutable/generator.hpp"


#ifndef DISABLE_GENERATOR_TEST


TEST(GENERATOR, Copy_J000)
{
	FUZZ::reset_logger();
	tensorshape shape = random_def_shape();
	double c = FUZZ::getDouble(1, "c", {1, 17})[0];

	std::vector<double> cdata(shape.n_elems(), 0);
	constant<double>* con = constant<double>::get(cdata, shape);
	const_init<double> cinit(c);
	rand_uniform<double> rinit(-12, -2);

	generator<double>* gen_assign = generator<double>::get(con, rinit);

	generator<double>* gen = generator<double>::get(con, cinit);

	generator<double>* gen_cpy = gen->clone();

	*gen_assign = *gen;

	EXPECT_TRUE(tensorshape_equal(gen->get_shape(), shape));
	EXPECT_TRUE(tensorshape_equal(gen_cpy->get_shape(), shape));
	EXPECT_TRUE(tensorshape_equal(gen_assign->get_shape(), shape));

	std::vector<double> gvec = nnet::expose<double>(gen);
	std::vector<double> gvec_cpy = nnet::expose<double>(gen_cpy);
	std::vector<double> gvec_assign = nnet::expose<double>(gen_assign);
	std::all_of(gvec.begin(), gvec.end(), [c](double e) { return e == c; });
	std::all_of(gvec_cpy.begin(), gvec_cpy.end(), [c](double e) { return e == c; });
	std::all_of(gvec_assign.begin(), gvec_assign.end(), [c](double e) { return e == c; });

	delete con;
}


TEST(GENERATOR, Move_J000)
{
	FUZZ::reset_logger();
	tensorshape shape = random_def_shape();
	double c = FUZZ::getDouble(1, "c", {1, 17})[0];

	std::vector<double> cdata(shape.n_elems(), 0);
	constant<double>* con = constant<double>::get(cdata, shape);
	const_init<double> cinit(c);
	rand_uniform<double> rinit(-12, -2);

	generator<double>* gen_assign = generator<double>::get(con, rinit);

	generator<double>* gen = generator<double>::get(con, cinit);
	EXPECT_TRUE(tensorshape_equal(gen->get_shape(), shape));
	std::vector<double> gvec = nnet::expose<double>(gen);
	std::all_of(gvec.begin(), gvec.end(), [c](double e) { return e == c; });

	generator<double>* gen_mv = gen->move();
	EXPECT_TRUE(tensorshape_equal(gen_mv->get_shape(), shape));
	std::vector<double> gvec_mv = nnet::expose<double>(gen_mv);
	std::all_of(gvec_mv.begin(), gvec_mv.end(), [c](double e) { return e == c; });

	*gen_assign = std::move(*gen_mv);
	EXPECT_TRUE(tensorshape_equal(gen_assign->get_shape(), shape));
	std::vector<double> gvec_assign = nnet::expose<double>(gen_assign);
	std::all_of(gvec_assign.begin(), gvec_assign.end(), [c](double e) { return e == c; });

	delete con;
	delete gen;
	delete gen_mv;
}


TEST(GENERATOR, ShapeDep_J001)
{
	FUZZ::reset_logger();
	tensorshape shape = random_def_shape();
	double c = FUZZ::getDouble(1, "c", {1, 17})[0];
	std::vector<double> cdata(shape.n_elems(), 0);
	constant<double>* con = constant<double>::get(cdata, shape);
	const_init<double> cinit(c);

	generator<double>* gen = generator<double>::get(con, cinit);
	EXPECT_TRUE(tensorshape_equal(gen->get_shape(), shape));

	delete con;
}


TEST(GENERATOR, Derive_J002)
{
	FUZZ::reset_logger();
	tensorshape shape = random_def_shape();
	double c = FUZZ::getDouble(1, "c", {1, 17})[0];
	std::vector<double> cdata(shape.n_elems(), 0);
	constant<double>* con = constant<double>::get(cdata, shape);
	const_init<double> cinit(c);

	generator<double>* gen = generator<double>::get(con, cinit);

	inode<double>* wan = nullptr;
	gen->temporary_eval(gen, wan);
	constant<double>* wanc = dynamic_cast<constant<double>*>(wan);
	ASSERT_NE(nullptr, wanc);
	EXPECT_TRUE(*wanc == 1.0);

	varptr<double> zaro = gen->derive(con);
	varptr<double> wan2 = gen->derive(gen);
	constant<double>* zaroc = dynamic_cast<constant<double>*>(zaro.get());
	ASSERT_NE(nullptr, zaroc);
	EXPECT_TRUE(*zaroc == 0.0);
	constant<double>* wanc2 = dynamic_cast<constant<double>*>(wan2.get());
	ASSERT_NE(nullptr, wanc2);
	EXPECT_TRUE(*wanc2 == 1.0);

	delete con;
	delete wan;
}


#endif /* DISABLE_GENERATOR_TEST */


#endif /* DISABLE_CONNECTOR_MODULE_TESTS */
