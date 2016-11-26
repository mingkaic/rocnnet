//
// Created by Mingkai Chen on 2016-08-29.
//

#include <limits>
#include "gtest/gtest.h"
#include "graph/operation/general/elementary.hpp"


#define UNARY std::function<nnet::varptr<double>(nnet::varptr<double>)>
#define BINARY_BOTH std::function<nnet::varptr<double>(nnet::varptr<double>, nnet::varptr<double>)>
#define BINARY_FIRST std::function<nnet::varptr<double>(nnet::varptr<double>, double)>
#define BINARY_SEC std::function<nnet::varptr<double>(double, nnet::varptr<double>)>

static const double epi = std::numeric_limits<double>::epsilon();


void print (std::vector<double> raw)
{
	for (double r : raw)
	{
		std::cout << r << " ";
	}
	std::cout << "\n";
}


void unaryElemTest (UNARY func,
	std::function<double(double)> expect_op)
{
	const size_t limit = 523;
	const size_t edge = 10;
	const size_t supersize = edge * edge * edge;
	nnet::placeholder<double> pin(std::vector<size_t>{edge, edge, edge}, "unar_in");

	// link in operation tree
	nnet::varptr<double> res = func(nnet::varptr<double>(&pin));

	// didn't initialize, but nothing interesting would happen
	// since we're using reactive
	nnet::expose<double>(res);

	// initialize
	std::vector<double> expect_out;
	for (size_t i = 0; i < supersize; i++)
	{
		double val = 0;
		while (0 == val)
		{
			val = fmod(rand(), limit);
		}
		expect_out.push_back(val);
	}
	pin = expect_out;

	// expose and test
	std::vector<double> raw = nnet::expose<double>(res);
	ASSERT_EQ(raw.size(), supersize);
	for (size_t i = 0; i < supersize; i++)
	{
		double err = std::abs(expect_op(expect_out[i]) - raw[i]);
		EXPECT_GE(epi, err);
	}
}

void binaryElemTest (
	BINARY_BOTH func, BINARY_FIRST func1, BINARY_SEC func2,
	std::function<double(double, double)> op)
{
	nnet::session& sess = nnet::session::get_instance();
	sess.enable_shape_eval();
	// should match shapes when first executing func

	const size_t limit = 523;
	const size_t edge = 10;
	const size_t badsize = edge * edge;
	const size_t supersize = edge * edge * edge;
	nnet::tensorshape goodshape = std::vector<size_t>{edge, edge, edge};
	nnet::placeholder<double> p1(goodshape, "bin_in1");
	nnet::placeholder<double> p2(goodshape, "bin_in2");
	nnet::placeholder<double> bad((std::vector<size_t>{edge, edge}), "bad");

	EXPECT_DEATH({func(nnet::varptr<double>(&p1), nnet::varptr<double>(&bad)); }, ".*");
	EXPECT_DEATH({func(nnet::varptr<double>(&bad), nnet::varptr<double>(&p1)); }, ".*");

	nnet::varptr<double> res = func(nnet::varptr<double>(&p1), nnet::varptr<double>(&p2));
	nnet::varptr<double> res1 = func1(nnet::varptr<double>(&p1), 2);
	nnet::varptr<double> res2 = func2(2, nnet::varptr<double>(&p1));

	// didn't initialize
	EXPECT_DEATH({ nnet::expose<double>(res); }, ".*");
	EXPECT_DEATH({ nnet::expose<double>(res1); }, ".*");
	EXPECT_DEATH({ nnet::expose<double>(res2); }, ".*");

	// initialize
	std::vector<double> expect1;
	std::vector<double> expect2;
	for (size_t i = 0; i < supersize; i++)
	{
		double val = 0;
		while (0 == val)
		{
			val = fmod(rand(), limit);
		}
		expect1.push_back(val);
		val = 0;
		while (0 == val)
		{
			val = fmod(rand(), limit);
		}
		expect2.push_back(val);
	}
	p1 = expect1;
	p2 = expect2;

	std::vector<double> raw = nnet::expose<double>(res);
	std::vector<double> raw1 = nnet::expose<double>(res1);
	std::vector<double> raw2 = nnet::expose<double>(res2);

	ASSERT_EQ(raw.size(), supersize);
	ASSERT_EQ(raw1.size(), supersize);
	ASSERT_EQ(raw2.size(), supersize);

	for (size_t i = 0; i < supersize; i++)
	{
		EXPECT_EQ(op(expect1[i], expect2[i]),
			raw[i]);
		EXPECT_EQ(op(expect1[i], 2),
			raw1[i]);
		EXPECT_EQ(op(2, expect1[i]),
			raw2[i]);
	}
	// prevent shape eval from leaking to other test
	sess.disable_shape_eval();
}


TEST(OPERATION, Abs)
{
	unaryElemTest(
	[](nnet::varptr<double> in) { return +in; },
	[](double var) { return +var; });
}


TEST(OPERATION, Neg)
{
	unaryElemTest(
	[](nnet::varptr<double> in) { return -in; },
	[](double var) { return -var; });
}


TEST(OPERATION, Sin)
{
	unaryElemTest(
	[](nnet::varptr<double> in) { return nnet::sin(in); },
	[](double var) { return sin(var); });
}


TEST(OPERATION, Cos)
{
	unaryElemTest(
	[](nnet::varptr<double> in) { return nnet::cos(in); },
	[](double var) { return cos(var); });

}


TEST(OPERATION, Tan)
{
	unaryElemTest(
	[](nnet::varptr<double> in) { return nnet::tan(in); },
	[](double var) { return tan(var); });
}


TEST(OPERATION, Csc)
{
	unaryElemTest(
	[](nnet::varptr<double> in) { return nnet::csc(in); },
	[](double var) { return 1/sin(var); });
}


TEST(OPERATION, Sec)
{
	unaryElemTest(
	[](nnet::varptr<double> in) { return nnet::sec(in); },
	[](double var) { return 1/cos(var); });
}


TEST(OPERATION, Cot)
{
	unaryElemTest(
	[](nnet::varptr<double> in) { return nnet::cot(in); },
	[](double var) { return cos(var)/sin(var); });
}


TEST(OPERATION, Rxp)
{
	unaryElemTest(
	[](nnet::varptr<double> in) { return nnet::exp(in); },
	[](double var) { return exp(var); });
}


// TEST(OPERATION, Root)
// {
// 	unaryElemTest(
// 	[](nnet::varptr<double> in) { return nnet::root(in); },
// 	[](double var) { return exp(var); });
// }


// TEST(OPERATION, Pow)
// {
// 	unaryElemTest(
// 	[](nnet::varptr<double> in) { return nnet::pow(in); },
// 	[](double var) { return exp(var); });
// }


TEST(OPERATION, ClipVal)
{
	double max = 92012;
	double min = -12415;
	unaryElemTest(
	[max, min](nnet::varptr<double> in) { return nnet::clip_val(in, min, max); },
	[max, min](double var)
	{
		if (var > max) var = max;
		else if (var < min) var = min;
		return var;
	});
}

//
//TEST(OPERATION, Add)
//{
//	binaryElemTest(
//	[](nnet::varptr<double> a, nnet::varptr<double> b) { return a+b; },
//	[](nnet::varptr<double> a, double b) { return a+b; },
//	[](double a, nnet::varptr<double> b) { return a+b; },
//	[](double a, double b) { return a+b; });
//}
//
//
//TEST(OPERATION, Sub)
//{
//	binaryElemTest(
//	[](nnet::varptr<double> a, nnet::varptr<double> b) { return a-b; },
//	[](nnet::varptr<double> a, double b) { return a-b; },
//	[](double a, nnet::varptr<double> b) { return a-b; },
//	[](double a, double b) { return a-b; });
//}
//
//
//TEST(OPERATION, Mul)
//{
//	binaryElemTest(
//	[](nnet::varptr<double> a, nnet::varptr<double> b) { return a * b; },
//	[](nnet::varptr<double> a, double b) { return a * b; },
//	[](double a, nnet::varptr<double> b) { return a * b; },
//	[](double a, double b) { return a * b; });
//}
//
//
//TEST(OPERATION, Div)
//{
//	binaryElemTest(
//	[](nnet::varptr<double> a, nnet::varptr<double> b) { return a/b; },
//	[](nnet::varptr<double> a, double b) { return a/b; },
//	[](double a, nnet::varptr<double> b) { return a/b; },
//	[](double a, double b) { return a/b; });
//}
