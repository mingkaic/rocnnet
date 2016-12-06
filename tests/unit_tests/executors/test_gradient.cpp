//
// Created by Mingkai Chen on 2016-08-29.
//

#include <limits>
#include "gtest/gtest.h"
#include "executor/gradient.hpp"
#include "graph/functions.hpp"
#include "graph/operation/transform.hpp"
#include "graph/operation/matmul.hpp"
#include "tensor_test_util.h"

static const double epi = std::numeric_limits<double>::epsilon();

static std::function<double(double)> sig = [](double x) {
	return 1 / (1 + std::exp(-x));
};

static std::function<double(double)> sig_prime = [](double x) {
	double s = sig(x);
	return s * (1 - s);
};

// DERIVATIVES

TEST(DERIVE, UnaryElementary)
{
	const size_t edge = 10;
	const size_t supersize = edge * edge * edge;
	nnet::random_uniform<double> rinit(0, 523);
	nnet::varptr<double> in = new nnet::variable<double>((std::vector<size_t>{edge, edge, edge}), rinit, "in");

	std::vector<nnet::varptr<double> > univars = {
		+in,
		-in,
		nnet::sin(in),
		nnet::cos(in),
		nnet::tan(in),
		nnet::csc(in),
		nnet::sec(in),
		nnet::cot(in),
		nnet::exp(in),
		// nnet::sqrt(in),
		// nnet::pow(in, 3),
		nnet::clip_val(in, 2.0, 4.0),
		nnet::clip_norm(in, 0.5),
	};

	std::vector<std::function<double(double)> > derivs = {
		[](double e) { return +1; },
		[](double e) { return -1; },
		[](double e) { return cos(e); },
		[](double e) { return -sin(e); },
		[](double e) { return 1.0/(cos(e) * cos(e)); },
		[](double e) { return -1.0/(sin(e)*tan(e)); },
		[](double e) { return tan(e)/cos(e); },
		[](double e) { return -(1.0/sin(e))*(1.0/sin(e)); },
		[](double e) { return exp(e); },
		// [](double e) { return 1/(2*std::sqrt(e)); },
		// [](double e) { return 3*e*e; },
		[](double e) { return 2; },
		[](double e) { return 0.5; },
	};

	size_t len = univars.size();
	// not part of TEST
	assert(len == derivs.size());

	static_cast<nnet::variable<double>*>(in.get())->initialize();
	std::vector<double> in_raw = nnet::expose<double>(in);
	for (size_t i = 0; i < len; i++)
	{
		nnet::gradient<double>* grad = new nnet::gradient<double>(univars[i], in);
		grad->freeze();
		grad->execute();
		size_t count = 0;
		std::vector<double> raw;
		grad->collect_grad([&count, &raw](nnet::ivariable<double>* key, nnet::placeholder<double>* value)
		{
			raw = nnet::expose<double>(value);
			count++;
		});
		ASSERT_EQ(1, count);

		for (size_t j = 0; j < raw.size(); j++)
		{
			EXPECT_EQ(derivs[i](in_raw[j]), raw[j]);
		}
		delete grad;
	}
	delete in.get();
}


TEST(DERIVE, BinaryElementary)
{
	const size_t edge = 10;
	const size_t supersize = edge*edge*edge;
	nnet::session& sess = nnet::session::get_instance();
	nnet::random_uniform<double> rinit(0, 523);
	nnet::varptr<double> a = new nnet::variable<double>((std::vector<size_t>{edge, edge, edge}), rinit, "a");
	nnet::varptr<double> b = new nnet::variable<double>((std::vector<size_t>{edge, edge, edge}), rinit, "b");
	nnet::varptr<double> bad = new nnet::variable<double>((std::vector<size_t>{edge, edge, edge}), rinit, "bad");

	std::vector<nnet::varptr<double> > univars = {
		a+b, a-b, a*b, a/b,
	};

	std::vector<std::function<double(double, double, bool)> > derivs = {
		[](double a, double b, bool is_a) { return 1; },
		[](double a, double b, bool is_a) { return is_a ? 1 : -1; },
		[](double a, double b, bool is_a) { return is_a ? b : a; },
		[](double a, double b, bool is_a) { return is_a ? 1/b : -a/(b*b); },
	};

	// not part of TEST
	assert(univars.size() == derivs.size());
	sess.initialize_all<double>();

	std::vector<double> ra = nnet::expose<double>(a);
	std::vector<double> rb = nnet::expose<double>(b);

	for (size_t i = 0; i < univars.size(); i++) {
		nnet::gradient<double>* grad = new nnet::gradient<double>(univars[i]);
		nnet::gradient<double>* bad_grad = new nnet::gradient<double>(univars[i], bad);

		grad->freeze();
		bad_grad->freeze();
		grad->execute();
		bad_grad->execute();
		size_t count = 0;
		std::vector<double> raw1;
		std::vector<double> raw2;
		grad->collect_grad([&count, &raw1, &raw2](nnet::ivariable<double>* key, nnet::placeholder<double>* value)
		{
			if (0 == count)
			{
				raw1 = nnet::expose<double>(value);
			}
			else
			{
				raw2 = nnet::expose<double>(value);
			}
			count++;
		});
		ASSERT_EQ(2, count);
		count = 0;
		bad_grad->collect_grad([&count](nnet::ivariable<double>* key, nnet::placeholder<double>* value)
		{
			// TODO: what should we do for bad gradients?
			count++;
		});
		EXPECT_EQ(1, count); // expect since we'll never check bad_grad anyways...

		ASSERT_EQ(raw1.size(), raw2.size());
		for (size_t j = 0; j < raw1.size(); j++) {
			double erra = std::abs(derivs[i](ra[j], rb[j], true) - raw1[j]);
			double errb = std::abs(derivs[i](ra[j], rb[j], false) - raw2[j]);
			// error allow slightly less stringent for binaries
			EXPECT_LE(erra, 100*epi);
			EXPECT_LE(errb, 100*epi);
		}
		delete grad;
	}
	delete a.get(); delete b.get(); delete bad.get();
}


TEST(DERIVE, Transform) {
	const size_t edge = 10;
	nnet::tensorshape commonshape = std::vector<size_t>{2, edge};
	nnet::session& sess = nnet::session::get_instance();
	nnet::random_uniform<double> rinit(0, 523);
	nnet::varptr<double> in = new nnet::variable<double>(commonshape, rinit, "in");

	std::vector<nnet::varptr<double> > univars = {
		nnet::transpose(in), // derivative has shape <1>
		nnet::fit(in, in), // very boring,
		nnet::extend(in, 0, 5), // derivative is expected to be shape <10>
		nnet::compress(in), // very boring too, consider change in the future
	};

	nnet::varptr<double> ex_grad_leaf = new nnet::variable<double>(1);
	std::vector<nnet::varptr<double> > derivs = {
		nnet::transpose(ex_grad_leaf),
		nnet::fit(ex_grad_leaf, in),
		nnet::extend(ex_grad_leaf, 0, 5),
		nnet::compress(ex_grad_leaf),
	};

	size_t len = univars.size();
	// not part of TEST
	assert(len == derivs.size());

	sess.initialize_all<double>();
	for (size_t i = 0; i < len; i++)
	{
		std::vector<double> ex_grad = nnet::expose<double>(derivs[i]);
		nnet::gradient<double>* grad = new nnet::gradient<double>(univars[i], in);
		grad->freeze();
		grad->execute();
		size_t count = 0;
		std::vector<double> raw;
		grad->collect_grad([&count, &raw](nnet::ivariable<double>* key, nnet::placeholder<double>* value)
		{
			raw = nnet::expose<double>(value);
			count++;
		});
		ASSERT_EQ(1, count);

		ASSERT_EQ(ex_grad.size(), raw.size());
		for (size_t j = 0; j < raw.size(); j++)
		{
			double err = std::abs(ex_grad[j] - raw[j]);
			EXPECT_LE(err, epi);
		}
		delete grad;
	}
	delete in.get(); delete ex_grad_leaf.get();
}


TEST(DERIVE, ComplexElementary) {
	const size_t edge = 10;
	const size_t supersize = edge*edge*edge;
	nnet::session& sess = nnet::session::get_instance();
	nnet::random_uniform<double> rinit(0, 523);
	nnet::varptr<double> p1 = new nnet::variable<double>((std::vector<size_t>{edge, edge, edge}), rinit, "p1");
	nnet::varptr<double> p2 = new nnet::variable<double>((std::vector<size_t>{edge, edge, edge}), rinit, "p2");

	sess.initialize_all<double>();
	nnet::varptr<double> o = nnet::sin(p1) + p1 * p2;

	std::vector<double> r1 = nnet::expose<double>(p1);
	std::vector<double> r2 = nnet::expose<double>(p2);

	nnet::gradient<double> grad(o);
	grad.freeze();
	grad.execute();
	std::vector<double> dp1;
	std::vector<double> dp2;
	size_t count = 0;
	grad.collect_grad(
	[&count, &dp1, &dp2, p1, p2](
		nnet::ivariable<double>* key,
		nnet::placeholder<double>* value)
	{
		if (0 == count)
		{
			dp1 = nnet::expose<double>(value);
		}
		else
		{
			dp2 = nnet::expose<double>(value);
		}
		count++;
	});

	// res = f(a,b) = sin(a)+a*b
	// df(a,b)/da = cos(a)+b
	// df(a,b)/db = a
	std::vector<double> raw = nnet::expose<double>(o);
	ASSERT_EQ(2, count);

	// expect the equation is correct
	for (size_t i = 0; i < raw.size(); i++) {
		EXPECT_EQ(sin(r1[i])+r1[i]*r2[i], raw[i]);
	}
	// assert derivative over p1 is correct
	for (size_t i = 0; i < dp1.size(); i++) {
		EXPECT_EQ(cos(r1[i])+r2[i], dp1[i]);
	}
	// assert derivative over p2 is correct
	for (size_t i = 0; i < dp2.size(); i++) {
		EXPECT_EQ(r1[i], dp2[i]);
	}
	delete p1;
	delete p2;
}


// tests deriving with respect to leaf (variable) nodes using sigmoid
TEST(DERIVE, SigmoidComplex) {
	const size_t edge = 10;
	const size_t supersize = edge*edge*edge;
	nnet::session& sess = nnet::session::get_instance();
	nnet::random_uniform<double> rinit(0, 523);
	nnet::varptr<double> x = new nnet::variable<double>((std::vector<size_t>{edge, edge, edge}), rinit, "p1");

	sess.initialize_all<double>();
	nnet::varptr<double> o = nnet::sigmoid(x);

	std::vector<double> xin = nnet::expose<double>(x);

	nnet::gradient<double> grad(o);
	grad.freeze();
	grad.execute();
	std::vector<double> der;
	grad.collect_grad(
	[&der](nnet::ivariable<double>* key, nnet::placeholder<double>* value)
	{
		der = nnet::expose<double>(value);
	});
	std::vector<double> raw = nnet::expose<double>(o);

	for (size_t i = 0; i < raw.size(); i++) {
		EXPECT_EQ(sig(xin[i]), raw[i]);
	}
	for (size_t i = 0; i < der.size(); i++) {
		double err = std::abs(sig_prime(xin[i]) - der[i]);
		EXPECT_LT(err, 0.0001);
	}
	delete x.get();
}


// tests deriving with respect to operation nodes
TEST(DERIVE, OperationDerive) {
	const size_t limit = 11;
	const size_t edge = 10;
	const size_t supersize = edge*edge;
	nnet::session& sess = nnet::session::get_instance();
	nnet::random_uniform<double> rinit(-1, limit);

	nnet::varptr<double> x = new nnet::variable<double>((std::vector<size_t>{edge, edge}), rinit, "p1");
	nnet::placeptr<double> place = new nnet::placeholder<double>(std::vector<size_t>{edge, edge}, "in");
	nnet::varptr<double> mul = x * place; // X * IN

	nnet::varptr<double> o = nnet::sigmoid(mul); // 1/(1+e^(-X * IN))
	nnet::gradient<double> grad(o, mul); // d(1/(1+e^(-X * IN))) / d(X * IN)
	grad.freeze();
	
	sess.initialize_all<double>();
	std::vector<double> placeholder_in;
	for (size_t i = 0; i < supersize; i++) {
		placeholder_in.push_back(fmod(rand(), limit));
	}
	*place = placeholder_in;

	std::vector<double> xin = nnet::expose<double>(mul);
	grad.execute();
	std::vector<double> der;
	grad.collect_grad(
	[&der](nnet::ivariable<double>* key, nnet::placeholder<double>* value)
	{
		der = nnet::expose<double>(value);
	});

	ASSERT_EQ(der.size(), xin.size());

	for (size_t i = 0; i < der.size(); i++) {
		// allow some errors since sig_prime and better ex are prone to rounding errors
		// sig_prime is a 2 step process:
		// 1. get sig which can cause rounding
		// 2. taken sig * (1 - sig) which can cause further rounding at 1 - sig
		EXPECT_LT(std::abs(sig_prime(xin[i]) - der[i]), 0.0001);
	}
	
	delete x.get(); delete place.get();
}


// TESTS TENSOR JACOBI
TEST(DERIVE, Matmul) {
	nnet::session& sess = nnet::session::get_instance();
	size_t m = 4, n = 5, k = 6;
	nnet::random_uniform<double> rinit(0, 523);
	nnet::varptr<double> A = new nnet::variable<double>((std::vector<size_t>{m, n}), rinit, "A");
	nnet::varptr<double> B = new nnet::variable<double>((std::vector<size_t>{k, m}), rinit, "B");
	nnet::varptr<double> C = nnet::matmul<double>::build(A, B);

	// TODO: simplify this pattern
	nnet::varptr<double> one = nnet::constant<double>::build(1);
	nnet::varptr<double> ex_grad_leaf = nnet::fit(one, C);

	// if A has shape <m, n> and B has shape <k, m>, then C has shape <k, n>
	// matmul(<k, n>, <k, m> ^ T) yields <m, n>
	nnet::varptr<double> expect_dA = nnet::matmul<double>::build(ex_grad_leaf, B, false, true);
	// matmul(<m, n> ^ T, <k, n>) yields <k, m>
	nnet::varptr<double> expect_dB = nnet::matmul<double>::build(A, ex_grad_leaf, true);

	sess.initialize_all<double>();

	// not part of test...
	std::vector<size_t> dAshape = expect_dA->get_shape().as_list();
	std::vector<size_t> dBshape = expect_dB->get_shape().as_list();
	assert(dAshape[0] == 4 && dAshape[1] == 5); // same as A
	assert(dBshape[0] == 6 && dBshape[1] == 4); // same as B

	// actually derive C
	nnet::gradient<double> grad(C);
	grad.freeze();
	grad.execute();
	std::vector<double> rawA;
	std::vector<double> rawB;
	size_t count = 0;
	grad.collect_grad([&count, A, B, &rawA, &rawB](nnet::ivariable<double>* key, nnet::placeholder<double>* value)
	{
		if (key == A.get())
		{
			rawA = nnet::expose<double>(value);
		}
		if (key == B.get())
		{
			rawB = nnet::expose<double>(value);
		}
		count++;
	});
	ASSERT_EQ(2, count);

	// verify against expected
	std::vector<double> exA = nnet::expose<double>(expect_dA);
	std::vector<double> exB = nnet::expose<double>(expect_dB);

	size_t asize = exA.size();
	ASSERT_EQ(asize, rawA.size());
	for (size_t i = 0; i < asize; i++)
	{
		EXPECT_EQ(exA[i], rawA[i]);
	}

	size_t bsize = exB.size();
	ASSERT_EQ(bsize, rawB.size());
	for (size_t i = 0; i < bsize; i++)
	{
		EXPECT_EQ(exB[i], rawB[i]);
	}

	delete A.get();
	delete B.get();
	delete one.get();
}


TEST(DERIVE, OpDeriveMatmul) {
	const size_t limit = 523;
	const size_t edge = 10;
	const size_t supersize = edge*edge;
	nnet::session& sess = nnet::session::get_instance();
	nnet::random_uniform<double> rinit(0, 523);

	nnet::varptr<double> x = new nnet::variable<double>((std::vector<size_t>{edge, edge}), rinit, "p1");
	nnet::placeptr<double> place = new nnet::placeholder<double>(std::vector<size_t>{edge, edge}, "in");
	nnet::varptr<double> mul = nnet::matmul<double>::build(x, nnet::varptr<double>(place.get())); // <X, IN>

	nnet::varptr<double> o = nnet::sigmoid(mul); // 1/(1+e^(-<X, IN>))
	nnet::gradient<double> grad(o, mul); // d(1/(1+e^(-X * IN))) / d(<X, IN>)
	sess.initialize_all<double>();

	std::vector<double> placeholder_in;
	for (size_t i = 0; i < supersize; i++) {
		placeholder_in.push_back(fmod(rand(), limit));
	}
	place = placeholder_in;

	std::vector<double> xin = nnet::expose<double>(mul);
	std::vector<double> der;
	grad.freeze();
	grad.execute();
	grad.collect_grad(
	[&der](nnet::ivariable<double>* key, nnet::placeholder<double>* value)
	{
		der = nnet::expose<double>(value);
	});

	ASSERT_EQ(der.size(), xin.size());

	for (size_t i = 0; i < der.size(); i++) {
		// allow some errors since sig_prime and better ex are prone to rounding errors
		// sig_prime is a 2 step process:
		// 1. get sig which can cause rounding
		// 2. taken sig * (1 - sig) which can cause further rounding at 1 - sig
		EXPECT_LT(std::abs(sig_prime(xin[i]) - der[i]), 0.0001);
	}

	delete x.get();
	delete place.get();
}


TEST(DERIVE, MatmulComplex)
{
	nnet::session& sess = nnet::session::get_instance();
 	size_t insize = 10;
 	size_t lay1 = 9;
 	size_t lay2 = 8;
 	nnet::random_uniform<double> rinit(-1, 1);
 	// simulate layers without bias (for simplicity)
 	// layer 1
 	nnet::placeptr<double> IN = new nnet::placeholder<double>(std::vector<size_t>{insize, 1}, "in");
 	nnet::varptr<double> W1 = new nnet::variable<double>((std::vector<size_t>{lay1, insize}), rinit, "w1");
 	nnet::varptr<double> layer1 = nnet::sigmoid<double>(nnet::matmul<double>::build(IN, W1)); // shape <lay1, 1>
 	// layer 2
 	nnet::varptr<double> W2 = new nnet::variable<double>((std::vector<size_t>{lay2, lay1}), rinit, "w2");
 	nnet::varptr<double> layer2 = nnet::sigmoid<double>(nnet::matmul<double>::build(layer1, W2)); // shape <lay2, 1>

	nnet::gradient<double> grad(layer2); // leaves are W1 and W2
	sess.initialize_all<double>();

	std::vector<double> inraw;
	for (size_t i = 0; i < insize; i++) {
		double val = fmod(rand(), 2);
		inraw.push_back(1 - val);
	}
	IN = inraw;

 	// expected grad:
	// wrt W1

	// wrt W2

	grad.freeze();
	grad.execute();
	std::vector<double> w1vec;
	std::vector<double> w2vec;
	grad.collect_grad(
	[W1, W2, &w1vec, &w2vec](nnet::ivariable<double>* key, nnet::placeholder<double>* value)
	{
		if (key == W1)
		{
			EXPECT_TRUE(tensorshape_equal(W1->get_shape(), value->get_shape()));
			w1vec = nnet::expose<double>(value);
		}
		if (key == W2)
		{
			EXPECT_TRUE(tensorshape_equal(W2->get_shape(), value->get_shape()));
			w2vec = nnet::expose<double>(value);
		}
	});

	// compare expected and real values
	delete IN.get();
	delete W1.get();
	delete W2.get();
}