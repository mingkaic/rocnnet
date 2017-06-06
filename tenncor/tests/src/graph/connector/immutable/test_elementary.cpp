//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_GRAPH_MODULE_TESTS

#include <algorithm>

#include "graph/leaf/variable.hpp"
#include "graph/connector/immutable/elementary.hpp"
#include "graph/varptr.hpp"

#include "gtest/gtest.h"
#include "util_test.h"
#include "fuzz.h"


#ifndef DISABLE_ELEMENTARY_TEST


using namespace nnet;


using UNARY_SCALAR = std::function<double(double)>;
using UNARY_VAR = std::function<varptr<double>(varptr<double>)>;
using BINARY_SCALARS = std::function<double(double, double)>;
using BINARY_VARS = std::function<varptr<double>(varptr<double>, varptr<double>)>;
using BINARY_VAR1 = std::function<varptr<double>(varptr<double>, double)>;
using BINARY_VAR2 = std::function<varptr<double>(double, varptr<double>)>;
using QUANARY_SCALARS = std::function<double(double, double, double, double)>;


static const double epi = std::numeric_limits<double>::epsilon();


// commonly used testing format
static void unaryElemTest (UNARY_VAR func,
	UNARY_SCALAR expect_forward, BINARY_SCALARS expect_back)
{
	tensorshape shape = random_def_shape();
	size_t inn = shape.n_elems();
	rand_uniform<double> rinit(2, 12);

	variable<double> var(shape, rinit, "unar_var");
	varptr<double> res = func(varptr<double>(&var));

	// Behavior J000
	EXPECT_EQ(nullptr, func(varptr<double>(nullptr)));

	// initialize
	var.initialize();
	std::vector<double> indata = expose<double>(&var);

	// compare data, shape must be equivalent, since we're testing elementary operations (Behavior J001)
	const tensor<double>* rawtens = res->get_eval();
	std::vector<double> rawf = rawtens->expose();
	ASSERT_TRUE(tensorshape_equal(shape, rawtens->get_shape()));
	ASSERT_EQ(rawf.size(), inn);
	for (size_t i = 0; i < inn; i++)
	{
		double rawd = rawf[i];
		double forwardd = expect_forward(indata[i]);
		// allow some error in case of rounding error
		double errf = std::abs(forwardd - rawd);
		EXPECT_GE(epi, errf);
	}

	const tensor<double>* backtens = res->get_gradient(&var)->get_eval();
	std::vector<double> rawb = backtens->expose();
	ASSERT_TRUE(tensorshape_equal(shape, backtens->get_shape()) || rawb.size() == 1);
	if (rawb.size() == 1)
	{
		double rawdb = rawb[0];
		double backd = expect_back(indata[0], 1);
		// allow some error in case of rounding error
		double errb = std::abs(backd - rawdb);
		EXPECT_GE(epi, errb);
	}
	else
	{
		for (size_t i = 0; i < inn; i++)
		{
			double rawdb = rawb[i];
			double backd = expect_back(indata[i], 1);
			// allow some error in case of rounding error
			double errb = std::abs(backd - rawdb);
			EXPECT_GE(epi, errb);
		}
	}
}

static void binaryElemTest (BINARY_VARS func, BINARY_VAR1 func1, BINARY_VAR2 func2,
	BINARY_SCALARS expect_forward, QUANARY_SCALARS expect_back)
{
	tensorshape shape = random_def_shape();
	size_t inn = shape.n_elems();
	rand_uniform<double> rinit(2, 12);

	std::vector<size_t> shapelist = shape.as_list();
	size_t mutate_idx = FUZZ::getInt(1, "mutate_idx", {0, shapelist.size()-1})[0];
	shapelist[mutate_idx]++;
	tensorshape shape2 = shapelist;

	// matching pair
	std::vector<double> scalars = FUZZ::getDouble(2, "scalars", {3, 50});
	variable<double> var(shape, rinit, "var");
	variable<double> var2(shape, rinit, "var2");

	// Behavior J000
	EXPECT_EQ(nullptr, func(varptr<double>(nullptr), varptr<double>(nullptr)));
	EXPECT_EQ(nullptr, func(varptr<double>(&var), varptr<double>(nullptr)));
	EXPECT_EQ(nullptr, func(varptr<double>(nullptr), varptr<double>(&var2)));

	variable<double> var3(shape2, rinit, "unmatching_in");
	varptr<double> res = func(varptr<double>(&var), varptr<double>(&var2));
	varptr<double> res1 = func1(varptr<double>(&var), scalars[1]);
	varptr<double> res2 = func2(scalars[0], varptr<double>(&var2));

	// initialize
	var.initialize();
	var2.initialize();
	var3.initialize();
	std::vector<double> indata = expose<double>(&var);
	std::vector<double> indata2 = expose<double>(&var2);

	// compare data, shape must be equivalent, since we're testing elementary operations (J001)
	const tensor<double>* tenn = res->get_eval();
	const tensor<double>* tenn1 = res1->get_eval();
	const tensor<double>* tenn2 = res2->get_eval();
	std::vector<double> raw = tenn->expose();
	std::vector<double> raw1 = tenn1->expose();
	std::vector<double> raw2 = tenn2->expose();
	ASSERT_TRUE(tensorshape_equal(shape, tenn->get_shape()));
	ASSERT_TRUE(tensorshape_equal(shape, tenn1->get_shape()));
	ASSERT_TRUE(tensorshape_equal(shape, tenn2->get_shape()));
	ASSERT_EQ(raw.size(), inn);
	ASSERT_EQ(raw1.size(), inn);
	ASSERT_EQ(raw2.size(), inn);
	for (size_t i = 0; i < inn; i++)
	{
		double rawd = raw[i];
		double rawd1 = raw1[i];
		double rawd2 = raw2[i];

		double forwardd = expect_forward(indata[i], indata2[i]);
		double forwardd1 = expect_forward(indata[i], scalars[1]);
		double forwardd2 = expect_forward(scalars[0], indata2[i]);

		// allow some error in case of rounding error
		double errf = std::abs(forwardd - rawd);
		double errf1 = std::abs(forwardd1 - rawd1);
		double errf2 = std::abs(forwardd2 - rawd2);
		EXPECT_GE(epi, errf);
		EXPECT_GE(epi, errf1);
		EXPECT_GE(epi, errf2);
	}

	const tensor<double>* backtens1 = res->get_gradient(&var)->get_eval();
	const tensor<double>* backtens2 = res->get_gradient(&var2)->get_eval();
	const tensor<double>* back1tens = res1->get_gradient(&var)->get_eval();
	const tensor<double>* back2tens = res2->get_gradient(&var2)->get_eval();
	std::vector<double> raw3 = backtens1->expose();
	std::vector<double> raw4 = backtens2->expose();
	std::vector<double> raw5 = back1tens->expose();
	std::vector<double> raw6 = back2tens->expose();
	ASSERT_TRUE(tensorshape_equal(shape, backtens1->get_shape()) || raw3.size() == 1);
	ASSERT_TRUE(tensorshape_equal(shape, backtens2->get_shape()) || raw4.size() == 1);
	ASSERT_TRUE(tensorshape_equal(shape, back1tens->get_shape()) || raw5.size() == 1);
	ASSERT_TRUE(tensorshape_equal(shape, back2tens->get_shape()) || raw6.size() == 1);
	if (raw3.size() == 1)
	{
		double rawd = raw3[0];
		double backd = expect_back(indata[0], indata2[0], 1, 0);
		// allow some error in case of rounding error
		double errb = std::abs(backd - rawd);
		EXPECT_GE(epi, errb);
	}
	if (raw4.size() == 1)
	{
		double rawd = raw4[0];
		double backd = expect_back(indata[0], indata2[0], 0, 1);
		// allow some error in case of rounding error
		double errb = std::abs(backd - rawd);
		EXPECT_GE(epi, errb);
	}
	if (raw5.size() == 1)
	{
		double rawd = raw5[0];
		double backd = expect_back(indata[0], scalars[1], 1, 0);
		// allow some error in case of rounding error
		double errb = std::abs(backd - rawd);
		EXPECT_GE(epi, errb);
	}
	if (raw6.size() == 1)
	{
		double rawd = raw6[0];
		double backd = expect_back(scalars[0], indata2[0], 0, 1);
		// allow some error in case of rounding error
		double errb = std::abs(backd - rawd);
		EXPECT_GE(epi, errb);
	}

	// Behavior J002
	EXPECT_THROW(func(varptr<double>(&var), varptr<double>(&var3)), std::exception);
	EXPECT_THROW(func(varptr<double>(&var3), varptr<double>(&var2)), std::exception);
}


TEST(ELEMENTARY, Abs_J000ToJ002)
{
	FUZZ::reset_logger();
	unaryElemTest(
	[](varptr<double> in) { return +in; },
	[](double var) { return +var; },
	[](double, double gvar) { return +gvar; });
}


TEST(ELEMENTARY, Neg_J000ToJ002)
{
	FUZZ::reset_logger();
	unaryElemTest(
	[](varptr<double> in) { return -in; },
	[](double var) { return -var; },
	[](double, double gvar) { return -gvar; });
}


TEST(ELEMENTARY, Sin_J000ToJ002)
{
	FUZZ::reset_logger();
	unaryElemTest(
	[](varptr<double> in) { return sin(in); },
	[](double var) { return sin(var); },
	[](double var, double gvar) { return gvar * cos(var); });
}


TEST(ELEMENTARY, Cos_J000ToJ002)
{
	FUZZ::reset_logger();
	unaryElemTest(
	[](varptr<double> in) { return cos(in); },
	[](double var) { return cos(var); },
	[](double var, double gvar) { return -gvar * sin(var); });

}


TEST(ELEMENTARY, Tan_J000ToJ002)
{
	FUZZ::reset_logger();
	unaryElemTest(
	[](varptr<double> in) { return tan(in); },
	[](double var) { return tan(var); },
	[](double var, double gvar)
	{
		double s = cos(var);
		return gvar / (s * s);
	});
}


TEST(ELEMENTARY, Csc_J000ToJ002)
{
	FUZZ::reset_logger();
	unaryElemTest(
	[](varptr<double> in) { return csc(in); },
	[](double var) { return 1/sin(var); },
	[](double var, double gvar)
	{
		return -gvar / (sin(var) * tan(var));
	});
}


TEST(ELEMENTARY, Sec_J000ToJ002)
{
	FUZZ::reset_logger();
	unaryElemTest(
	[](varptr<double> in) { return sec(in); },
	[](double var) { return 1/cos(var); },
	[](double var, double gvar) { return gvar * tan(var) / cos(var); });
}


TEST(ELEMENTARY, Cot_J000ToJ002)
{
	FUZZ::reset_logger();
	unaryElemTest(
	[](varptr<double> in) { return cot(in); },
	[](double var) { return cos(var) / sin(var); },
	[](double var, double gvar)
	{
		double c = 1/sin(var);
		return -gvar * c * c;
	});
}


TEST(ELEMENTARY, Exp_J000ToJ002)
{
	FUZZ::reset_logger();
	unaryElemTest(
	[](varptr<double> in) { return exp(in); },
	[](double var) { return exp(var); },
	[](double var, double gvar) { return gvar * exp(var); });
}


// TEST(ELEMENTARY, Root_J000ToJ002)
// {
//FUZZ::reset_logger();
// 	unaryElemTest(
// 	[](varptr<double> in) { return root(in); },
// 	[](double var) { return exp(var); });
// }


TEST(ELEMENTARY, ClipVal_J000ToJ002)
{
	FUZZ::reset_logger();
	std::vector<double> limits = FUZZ::getDouble(2, "limits", {-100, 200});
	double min = limits[0] > limits[1] ? limits[1] : limits[0];
	double max = limits[0] > limits[1] ? limits[0] : limits[1];
	unaryElemTest(
	[max, min](varptr<double> in) { return clip_val(in, min, max); },
	[max, min](double var)
	{
		if (var > max) var = max;
		else if (var < min) var = min;
		return var;
	},
	[max, min](double var, double gvar)
	{
		if (var > max) var = max;
		else if (var < min) var = min;
		return gvar * var;
	});
}


TEST(ELEMENTARY, Add_J000ToJ003)
{
	FUZZ::reset_logger();
	binaryElemTest(
	[](varptr<double> a, varptr<double> b) { return a+b; },
	[](varptr<double> a, double b) { return a+b; },
	[](double a, varptr<double> b) { return a+b; },
	[](double a, double b) { return a+b; },
	[](double, double, double ga, double gb) { return ga+gb; });

	tensorshape shape = random_def_shape();
	rand_uniform<double> rinit(2, 12);
	varptr<double> zero = constant<double>::get(0.0);
	variable<double> var(shape, rinit, "var");
	variable<double> var2(shape, rinit, "var2");

	// Behavior J003
	varptr<double> samev1 = varptr<double>(&var) +  0.0;
	varptr<double> samev2 = 0.0 + varptr<double>(&var2);
	varptr<double> samev12 = varptr<double>(&var) + varptr<double>(zero);
	varptr<double> samev22 = varptr<double>(zero) + varptr<double>(&var2);

	EXPECT_EQ(&var, samev1.get());
	EXPECT_EQ(&var2, samev2.get());
	EXPECT_EQ(&var, samev12.get());
	EXPECT_EQ(&var2, samev22.get());

	// never consumed by a node
	delete zero;
}


TEST(ELEMENTARY, Sub_J000ToJ002_J004)
{
	FUZZ::reset_logger();
	binaryElemTest(
	[](varptr<double> a, varptr<double> b) { return a-b; },
	[](varptr<double> a, double b) { return a-b; },
	[](double a, varptr<double> b) { return a-b; },
	[](double a, double b) { return a-b; },
	[](double, double, double ga, double gb) { return ga-gb; });

	tensorshape shape = random_def_shape();
	size_t inn = shape.n_elems();
	rand_uniform<double> rinit(2, 12);
	varptr<double> zero = constant<double>::get(0.0);
	variable<double> var(shape, rinit, "var");
	variable<double> var2(shape, rinit, "var2");

	// Behavior J004
	varptr<double> samev1 = varptr<double>(&var) -  0.0;
	varptr<double> samenv2 = 0.0 - varptr<double>(&var2);
	varptr<double> samev12 = varptr<double>(&var) - varptr<double>(zero);
	varptr<double> samenv22 = varptr<double>(zero) - varptr<double>(&var2);

	EXPECT_EQ(&var, samev1.get());
	EXPECT_EQ(&var, samev12.get());

	// initialize
	var2.initialize();
	std::vector<double> indata2 = expose<double>(&var2);

	const tensor<double>* rawtens = samenv2->get_eval();
	const tensor<double>* rawtens2 = samenv22->get_eval();
	std::vector<double> rawf = rawtens->expose();
	std::vector<double> rawf2 = rawtens2->expose();
	ASSERT_TRUE(tensorshape_equal(shape, rawtens->get_shape()));
	ASSERT_TRUE(tensorshape_equal(shape, rawtens2->get_shape()));
	ASSERT_EQ(rawf.size(), inn);
	ASSERT_EQ(rawf2.size(), inn);
	for (size_t i = 0; i < inn; i++)
	{
		EXPECT_EQ(-indata2[i], rawf[i]);
		EXPECT_EQ(-indata2[i], rawf2[i]);
	}

	// never consumed by a node
	delete zero;
}


TEST(ELEMENTARY, Mul_J000ToJ002_J005ToJ006)
{
	FUZZ::reset_logger();
	binaryElemTest(
	[](varptr<double> a, varptr<double> b) { return a * b; },
	[](varptr<double> a, double b) { return a * b; },
	[](double a, varptr<double> b) { return a * b; },
	[](double a, double b) { return a * b; },
	[](double a, double b, double ga, double gb) { return ga*b+gb*a; });

	tensorshape shape = random_def_shape();
	rand_uniform<double> rinit(2, 12);
	varptr<double> zero = constant<double>::get(0.0);
	varptr<double> one = constant<double>::get(1.0);
	variable<double> var(shape, rinit, "var");
	variable<double> var2(shape, rinit, "var2");

	// Behavior J005
	varptr<double> zaro = varptr<double>(&var) *  0.0;
	varptr<double> zaro2 = 0.0 * varptr<double>(&var2);
	varptr<double> zaro3 = varptr<double>(&var) * varptr<double>(zero);
	varptr<double> zaro4 = varptr<double>(zero) * varptr<double>(&var2);
	std::vector<double> exp01 = expose<double>(zaro);
	std::vector<double> exp02 = expose<double>(zaro2);
	std::vector<double> exp03 = expose<double>(zaro3);
	std::vector<double> exp04 = expose<double>(zaro4);
	ASSERT_EQ((size_t) 1, exp01.size());
	ASSERT_EQ((size_t) 1, exp02.size());
	ASSERT_EQ((size_t) 1, exp03.size());
	ASSERT_EQ((size_t) 1, exp04.size());
	EXPECT_EQ(0, exp01.at(0));
	EXPECT_EQ(0, exp02.at(0));
	EXPECT_EQ(0, exp03.at(0));
	EXPECT_EQ(0, exp04.at(0));

	// Behavior J006
	varptr<double> samev1 = varptr<double>(&var) * 1.0;
	varptr<double> samev2 = 1.0 * varptr<double>(&var2);
	varptr<double> samev12 = varptr<double>(&var) * varptr<double>(one);
	varptr<double> samev22 = varptr<double>(one) * varptr<double>(&var2);

	EXPECT_EQ(&var, samev1.get());
	EXPECT_EQ(&var2, samev2.get());
	EXPECT_EQ(&var, samev12.get());
	EXPECT_EQ(&var2, samev22.get());

	// never consumed by a node
	delete zero;
	delete one;
}


TEST(ELEMENTARY, Div_J000ToJ002_J007ToJ008)
{
	FUZZ::reset_logger();
	binaryElemTest(
	[](varptr<double> a, varptr<double> b) { return a/b; },
	[](varptr<double> a, double b) { return a/b; },
	[](double a, varptr<double> b) { return a/b; },
	[](double a, double b) { return a/b; },
	[](double a, double b, double ga, double gb) { return (ga*b-gb*a)/(b*b); });

	tensorshape shape = random_def_shape();
	rand_uniform<double> rinit(2, 12);
	varptr<double> zero = constant<double>::get(0.0);
	varptr<double> one = constant<double>::get(1.0);
	variable<double> var(shape, rinit, "var");
	variable<double> var2(shape, rinit, "var2");

	// Behavior J005
	varptr<double> zaro = 0.0 / varptr<double>(&var2);
	varptr<double> zaro2 = varptr<double>(zero) * varptr<double>(&var2);
	EXPECT_THROW(varptr<double>(&var) /  0.0, std::logic_error);
	EXPECT_THROW(varptr<double>(&var) / varptr<double>(zero), std::logic_error);

	std::vector<double> exp01 = expose<double>(zaro);
	std::vector<double> exp02 = expose<double>(zaro2);
	ASSERT_EQ((size_t) 1, exp01.size());
	ASSERT_EQ((size_t) 1, exp02.size());
	EXPECT_EQ(0, exp01.at(0));
	EXPECT_EQ(0, exp02.at(0));

	// Behavior J006
	varptr<double> samev1 = varptr<double>(&var) / 1.0;
	varptr<double> samev12 = varptr<double>(&var) / varptr<double>(one);

	EXPECT_EQ(&var, samev1.get());
	EXPECT_EQ(&var, samev12.get());

	// never consumed by a node
	delete zero;
	delete one;
}


// TEST(ELEMENTARY, Pow)
// {
//	FUZZ::reset_logger();
// 	unaryElemTest(
// 	[](varptr<double> in) { return pow(in); },
// 	[](double var) { return exp(var); });
// }


#endif /* DISABLE_ELEMENTARY_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */
