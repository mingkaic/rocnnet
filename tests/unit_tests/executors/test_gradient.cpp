////
//// Created by Mingkai Chen on 2016-08-29.
////
//
//#include <limits>
//#include "gtest/gtest.h"
//#include "executor/gradient.hpp"
//#include "graph/functions.hpp"
//
//static const double epi = std::numeric_limits<double>::epsilon();
//
//// DERIVATIVES
//
//TEST(DERIVE, unary) {
//	const size_t edge = 10;
//	const size_t supersize = edge * edge * edge;
//	nnet::random_uniform<double> rinit(0, 523);
//	nnet::varptr<double> in = new nnet::variable<double>((std::vector<size_t>{edge, edge, edge}), rinit, "in");
//
//	std::vector<nnet::varptr<double> > univars = {
//		+in,
//		-in,
//		nnet::sin(in),
//		nnet::cos(in),
//		nnet::tan(in),
//		nnet::csc(in),
//		nnet::sec(in),
//		nnet::cot(in),
//		nnet::exp(in),
//	};
//
//	std::vector<std::function<double(double)> > derivs = {
//		[](double e) { return +1; },
//		[](double e) { return -1; },
//		[](double e) { return cos(e); },
//		[](double e) { return -sin(e); },
//		[](double e) { return 1.0/(cos(e) * cos(e)); },
//		[](double e) { return -1.0/(sin(e)*tan(e)); },
//		[](double e) { return tan(e)/cos(e); },
//		[](double e) { return -(1.0/sin(e))*(1.0/sin(e)); },
//		[](double e) { return exp(e); },
//	};
//
//	size_t len = univars.size();
//	// not part of TEST
//	assert(len == derivs.size());
//
//    static_cast<nnet::variable<double>*>(in.get())->initialize();
//    std::vector<double> in_raw = nnet::expose<double>(in);
//	for (size_t i = 0; i < len; i++)
//	{
//		nnet::gradient<double>* grad = new nnet::gradient<double>(univars[i], in);
//		grad->freeze();
//		size_t count = 0;
//		std::vector<double> raw;
//		grad->collect_grad([&count, &raw](nnet::ivariable<double>* key, nnet::placeholder<double>* value)
//		{
//		    raw = nnet::expose<double>(value);
//		    count++;
//		});
//		ASSERT_EQ(1, count);
//
//		for (size_t j = 0; j < raw.size(); j++)
//		{
//			EXPECT_EQ(derivs[i](in_raw[j]), raw[j]);
//		}
//		delete grad;
//	}
//	delete in.get();
//}
//
//
//TEST(DERIVE, binary) {
//	const size_t edge = 10;
//	const size_t supersize = edge*edge*edge;
//	nnet::session& sess = nnet::session::get_instance();
//	nnet::random_uniform<double> rinit(0, 523);
//	nnet::varptr<double> a = new nnet::variable<double>((std::vector<size_t>{edge, edge, edge}), rinit, "a");
//	nnet::varptr<double> b = new nnet::variable<double>((std::vector<size_t>{edge, edge, edge}), rinit, "b");
//	nnet::varptr<double> bad = new nnet::variable<double>((std::vector<size_t>{edge, edge, edge}), rinit, "bad");
//
//	std::vector<nnet::varptr<double> > univars = {
//		a+b, a-b, a*b, a/b,
//	};
//
//	std::vector<std::function<double(double, double, bool)> > derivs = {
//		[](double a, double b, bool is_a) { return 1; },
//		[](double a, double b, bool is_a) { return is_a ? 1 : -1; },
//		[](double a, double b, bool is_a) { return is_a ? b : a; },
//		[](double a, double b, bool is_a) { return is_a ? 1/b : -a/(b*b); },
//	};
//
//	// not part of TEST
//	assert(univars.size() == derivs.size());
//	sess.initialize_all<double>();
//
//	std::vector<double> ra = nnet::expose<double>(a);
//	std::vector<double> rb = nnet::expose<double>(b);
//
//	for (size_t i = 0; i < univars.size(); i++) {
//		nnet::gradient<double>* grad = new nnet::gradient<double>(univars[i]);
//		nnet::gradient<double>* bad_grad = new nnet::gradient<double>(univars[i], bad);
//
//		grad->freeze();
//		bad_grad->freeze();
//		size_t count = 0;
//		std::vector<double> raw1;
//		std::vector<double> raw2;
//		grad->collect_grad([&count, &raw1, &raw2](nnet::ivariable<double>* key, nnet::placeholder<double>* value)
//		{
//		    if (0 == count)
//    		    raw1 = nnet::expose<double>(value);
//    		else
//    		    raw2 = nnet::expose<double>(value);
//		    count++;
//		});
//		ASSERT_EQ(2, count);
//		count = 0;
//		bad_grad->collect_grad([&count](nnet::ivariable<double>* key, nnet::placeholder<double>* value)
//		{
//		    count++;
//		});
//		ASSERT_EQ(0, count);
//
//		ASSERT_EQ(raw1.size(), raw2.size());
//		for (size_t j = 0; j < raw1.size(); j++) {
//			double erra = std::abs(derivs[i](ra[j], rb[j], true) - raw1[j]);
//			double errb = std::abs(derivs[i](ra[j], rb[j], false) - raw2[j]);
//			EXPECT_LE(erra, epi);
//			EXPECT_LE(errb, epi);
//		}
//	}
//}
//
//
//// TODO: write these
//// TEST(DERIVE, transpose) {
//
//// }
//
//
//// TEST(DERIVE, matop) {
//
//// }
//
//
//// TEST(DERIVE, complex) {
//// 	const size_t edge = 10;
//// 	const size_t supersize = edge*edge*edge;
//// 	nnet::session& sess = nnet::session::get_instance();
//// 	nnet::random_uniform<double> rinit(0, 523);
//// 	nnet::varptr<double> p1 = new nnet::variable<double>((std::vector<size_t>{edge, edge, edge}), rinit, "p1");
//// 	nnet::varptr<double> p2 = new nnet::variable<double>((std::vector<size_t>{edge, edge, edge}), rinit, "p2");
//
//// 	sess.initialize_all<double>();
//// 	nnet::varptr<double> o = nnet::sin(p1) + p1 * p2;
//// 	nnet::expose<double>* res = new nnet::expose<double>(o);
//
//// 	std::vector<double> r1 = new nnet::expose<double>(p1)->get_raw();
//// 	std::vector<double> r2 = new nnet::expose<double>(p2)->get_raw();
//
//// 	nnet::varptr<double> grad1 = new nnet::gradient<double>(res, p1);
//// 	nnet::varptr<double> grad2 = new nnet::gradient<double>(res, p2);
//
//// 	// res = f(a,b) = sin(a)+a*b
//// 	// df(a,b)/da = cos(a)+b
//// 	// df(a,b)/db = a
//// 	std::vector<double> raw = res->get_raw();
//// 	std::vector<double> dp1 = new nnet::expose<double>(grad1)->get_raw();
//// 	std::vector<double> dp2 = new nnet::expose<double>(grad2)->get_raw();
//
//// 	for (size_t i = 0; i < raw.size(); i++) {
//// 		EXPECT_EQ(sin(r1[i])+r1[i]*r2[i], raw[i]);
//// 	}
//// 	for (size_t i = 0; i < dp1.size(); i++) {
//// 		EXPECT_EQ(cos(r1[i])+r2[i], dp1[i]);
//// 	}
//// 	for (size_t i = 0; i < dp2.size(); i++) {
//// 		EXPECT_EQ(r1[i], dp2[i]);
//// 	}
//// }
//
//
//// // tests deriving with respect to leaf (variable) nodes using sigmoid
//// TEST(DERIVE, sigmoid_complex) {
//// 	const size_t edge = 10;
//// 	const size_t supersize = edge*edge*edge;
//// 	nnet::session& sess = nnet::session::get_instance();
//// 	nnet::random_uniform<double> rinit(0, 523);
//// 	nnet::varptr<double> x = new nnet::variable<double>((std::vector<size_t>{edge, edge, edge}), rinit, "p1");
//
//// 	sess.initialize_all<double>();
//// 	nnet::varptr<double> o = nnet::sigmoid(x);
//// 	nnet::expose<double>* res = new nnet::expose<double>(o);
//
//// 	std::vector<double> xin = new nnet::expose<double>(x)->get_raw();
//
//// 	std::function<double(double)> sig = [](double x) {
//// 		return 1 / (1 + std::exp(-x));
//// 	};
//// 	std::function<double(double)> sig_prime = [&sig](double x) {
//// 		double s = sig(x);
//// 		return s * (1 - s);
//// 	};
//
//// 	nnet::varptr<double> grad = new nnet::ivariable<double>(res, x);
//// 	std::vector<double> raw = res->get_raw();
//// 	std::vector<double> der = new nnet::expose<double>(grad)->get_raw();;
//
//// 	for (size_t i = 0; i < raw.size(); i++) {
//// 		EXPECT_EQ(sig(xin[i]), raw[i]);
//// 	}
//// 	for (size_t i = 0; i < der.size(); i++) {
//// 		double err = std::abs(sig_prime(xin[i]) - der[i]);
//// 		EXPECT_LT(err, 0.0001);
//// 	}
//// }
//
//
//// // tests deriving with respect to operation nodes
//// TEST(DERIVE, operation_derive) {
//// 	const size_t limit = 523;
//// 	const size_t edge = 10;
//// 	const size_t supersize = edge*edge;
//// 	nnet::session& sess = nnet::session::get_instance();
//// 	nnet::random_uniform<double> rinit(0, 523);
//
//// 	nnet::varptr<double> x = new nnet::variable<double>((std::vector<size_t>{edge, edge}), rinit, "p1");
//// 	nnet::placeptr<double> place = new nnet::placeholder<double>(std::vector<size_t>{edge, edge}, "in");
//// 	nnet::varptr<double> mul = new nnet::matmul<double>(x, place); // <X, IN>
//
//// 	nnet::varptr<double> o = nnet::sigmoid(mul); // 1/(1+e^(-<X, IN>))
//// 	nnet::varptr<double> better_grad = o*(1.0-o); // d(1/(1+e^(-<X, IN>))) / d(<X, IN>)
//// 	nnet::varptr<double> grad = new nnet::derive<double>(o, mul); // d(1/(1+e^(-<X, IN>))) / d(<X, IN>)
//// 	nnet::expose<double>* oex = new nnet::expose<double>(o);
//// 	nnet::expose<double>* ex = new nnet::expose<double>(grad);
//// 	nnet::expose<double>* better_ex = new nnet::expose<double>(better_grad);
//// 	sess.initialize_all<double>();
//
//// 	std::vector<double> placeholder_in;
//// 	for (size_t i = 0; i < supersize; i++) {
//// 		placeholder_in.push_back(fmod(rand(), limit));
//// 	}
//// 	*place = placeholder_in;
//
//// 	std::vector<double> xin = new nnet::expose<double>(mul)->get_raw();
//// 	std::function<double(double)> sig = [](double x) {
//// 		return 1 / (1 + std::exp(-x));
//// 	};
//// 	std::function<double(double)> sig_prime = [&sig](double x) {
//// 		double s = sig(x);
//// 		return s * (1 - s);
//// 	};
//
//// 	std::vector<double> der = ex->get_raw();
//// 	std::vector<double> raw = better_ex->get_raw();
//
//// 	ASSERT_EQ(der.size(), xin.size());
//// 	ASSERT_EQ(der.size(), raw.size());
//
//// 	for (size_t i = 0; i < der.size(); i++) {
//// 		// allow some errors since sig_prime and better ex are prone to rounding errors
//// 		// sig_prime is a 2 step process:
//// 		// 1. get sig which can cause rounding
//// 		// 2. taken sig * (1 - sig) which can cause further rounding at 1 - sig
//// 		EXPECT_LT(std::abs(sig_prime(xin[i]) - der[i]), 0.0001);
//// 		EXPECT_LT(std::abs(raw[i] - der[i]), 0.0001);
//// 	}
//// }