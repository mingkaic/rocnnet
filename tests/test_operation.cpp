//
//  test_operation.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <stdio.h>
#include <algorithm>
#include "../include/graph.hpp"

template <typename T>
void unaryElemTest (
	std::function<double(double)> op) {
	const size_t edge = 10;
	const size_t supersize = edge*edge*edge;
	nnet::placeholder<double> p(std::vector<size_t>{edge, edge, edge});

	T res(p);

	// didn't initialize
	EXPECT_DEATH({ res.eval(); }, ".*");

	std::vector<double> r;
	for (size_t i = 0; i < supersize; i++) {
		r.push_back(rand());
	}
	p = r;

	nnet::expose<double> ex(res);
	// evaluates
    std::vector<double> raw = ex.get_raw();

    ASSERT_EQ(raw.size(), supersize);

    for (size_t i = 0; i < supersize; i++) {
		EXPECT_EQ(op(r[i]), raw[i]);
    }
}

template <typename T>
void binaryElemTest (
	std::function<double(double, double)> op) {
	const size_t edge = 10;
	const size_t badsize = edge*edge;
	const size_t supersize = edge*edge*edge;
	nnet::tensor_shape goodshape = std::vector<size_t>{edge, edge, edge};
	nnet::placeholder<double> p1(goodshape);
	nnet::placeholder<double> p2(goodshape);
	nnet::placeholder<double> bad(std::vector<size_t>{edge, edge});

	T trouble1(p1, bad);
	T trouble2(bad, p1);
	T res(p1, p2);
	T res1(p1, 2);
	T _1res(2, p1);

	// didn't initialize
	EXPECT_DEATH({ trouble1.eval(); }, ".*");
	EXPECT_DEATH({ trouble2.eval(); }, ".*");
	EXPECT_DEATH({ res.eval(); }, ".*");
	EXPECT_DEATH({ res1.eval(); }, ".*");
	EXPECT_DEATH({ _1res.eval(); }, ".*");

	// initialize
	std::vector<double> r1;
	std::vector<double> r2;
	std::vector<double> rbad;
	for (size_t i = 0; i < supersize; i++) {
		r1.push_back(rand());
		r2.push_back(rand());
	}
	for (size_t i = 0; i < badsize; i++) {
		rbad.push_back(rand());
	}
	p1 = r1;
	p2 = r2;
	bad = rbad;

	// delayed evaluation
	EXPECT_THROW({ trouble1.eval(); }, std::invalid_argument);
	EXPECT_THROW({ trouble2.eval(); }, std::invalid_argument);

	nnet::expose<double> ex(res);
	nnet::expose<double> ex1(res1);
	nnet::expose<double> ex2(_1res);
	// evaluates
    std::vector<double> raw = ex.get_raw();
	std::vector<double> raw1 = ex1.get_raw();
	std::vector<double> raw2 = ex2.get_raw();

    ASSERT_EQ(raw.size(), supersize);
	ASSERT_EQ(raw1.size(), supersize);
	ASSERT_EQ(raw2.size(), supersize);

    for (size_t i = 0; i < supersize; i++) {
		EXPECT_EQ(op(r1[i], r2[i]), raw[i]);
		EXPECT_EQ(op(r1[i], 2), raw1[i]);
		EXPECT_EQ(op(2, r1[i]), raw2[i]);
    }
}


TEST(OPERATION, neg) {
	unaryElemTest<nnet::neg<double> >([](double var) {
		return -var;
	});
}


TEST(OPERATION, sin) {
	unaryElemTest<nnet::sin<double> >([](double var) {
		return sin(var);
	});
}


TEST(OPERATION, cos) {
	unaryElemTest<nnet::cos<double> >([](double var) {
		return cos(var);
	});

}


TEST(OPERATION, tan) {
	unaryElemTest<nnet::tan<double> >([](double var) {
		return tan(var);
	});
}


TEST(OPERATION, csc) {
	unaryElemTest<nnet::csc<double> >([](double var) {
		return 1/sin(var);
	});
}


TEST(OPERATION, sec) {
	unaryElemTest<nnet::sec<double> >([](double var) {
		return 1/cos(var);
	});
}


TEST(OPERATION, cot) {
	unaryElemTest<nnet::cot<double> >([](double var) {
		return cos(var)/sin(var);
	});
}


TEST(OPERATION, scalar) {
	nnet::scalar<double> zero(0);
	nnet::scalar<double> uno(1);
	nnet::scalar<double> one(1);
	nnet::scalar<double> deux(2);

	uno.derive(&one);
}


TEST(OPERATION, exp) {
	unaryElemTest<nnet::exp<double> >([](double var) {
		return exp(var);
	});
}


TEST(OPERATION, add) {
	binaryElemTest<nnet::add<double> >([](double a, double b) {
		return a+b;
	});
}


TEST(OPERATION, sub) {
	binaryElemTest<nnet::sub<double> >([](double a, double b) {
		return a-b;
	});
}


TEST(OPERATION, mul) {
	binaryElemTest<nnet::mul<double> >([](double a, double b) {
		return a*b;
	});
}


TEST(OPERATION, div) {
	binaryElemTest<nnet::div<double> >([](double a, double b) {
		return a/b;
	});
}


TEST(OPERATION, matmul) {
	const size_t limit = 523;
	const size_t ncol = 3;
	const size_t nrow = 4;
	std::vector<double> av = {
		3, 4, 5,
		41, 6, 7,
		8, 1, 9,
		18, 2, 0
	};
	std::vector<double> bv = {
		2, 3, 10,
		12, 54, 11,
		0, 9, 0,
		0.1, 29, 0
	};
	std::vector<double> cv = {
		11, 22, 32, 9,
		6, 4, 45, 3.2,
		6, 3, 3, 3
	};
	double ex1[4][4] = {
		{68, 307, 36, 116.3},
		{170, 893, 54, 178.1},
		{109, 249, 9, 29.8},
		{42, 324, 18, 59.8}
	};
	double ex2[3][3] = {
		{499.8, 2817, 481},
		{80.2, 403, 106},
		{94, 474, 127}
	};
	double ex3[3][3] = {
		{1353, 226, 497},
		{599.6, 99.4, 463},
		{219, 51, 78}
	};
	const size_t supersize = ncol*nrow;
	nnet::placeholder<double> A(std::vector<size_t>{ncol, nrow});
	nnet::placeholder<double> B(std::vector<size_t>{ncol, nrow});
	nnet::placeholder<double> C(std::vector<size_t>{nrow, ncol});
	nnet::matmul<double> ans1(A, B, false, true); // output is 4x4
	nnet::matmul<double> ans2(A, B, true); // output is 3x3
	nnet::matmul<double> ans3(C, A); // output is 3x3

	// didn't initialize
	EXPECT_DEATH({ ans1.eval(); }, ".*");
	EXPECT_DEATH({ ans2.eval(); }, ".*");
	EXPECT_DEATH({ ans3.eval(); }, ".*");
	A = av;
	B = bv;
	C = cv;

	nnet::expose<double> res1(ans1);
	nnet::expose<double> res2(ans2);
	nnet::expose<double> res3(ans3);
	// evaluates
    nnet::tensor<double> t1 = res1.eval();
	nnet::tensor_shape s1 = t1.get_shape();
	std::vector<size_t> v1 = s1.as_list();
	ASSERT_EQ(v1.size(), 2);
	ASSERT_EQ(v1[0], nrow);
	ASSERT_EQ(v1[1], nrow);

    nnet::tensor<double> t2 = res2.eval();
	nnet::tensor_shape s2 = t2.get_shape();
	std::vector<size_t> v2 = s2.as_list();
	ASSERT_EQ(v2.size(), 2);
	ASSERT_EQ(v2[0], ncol);
	ASSERT_EQ(v2[1], ncol);

    nnet::tensor<double> t3 = res3.eval();
	nnet::tensor_shape s3 = t3.get_shape();
	std::vector<size_t> v3 = s3.as_list();
	ASSERT_EQ(v3.size(), 2);
	ASSERT_EQ(v3[0], ncol);
	ASSERT_EQ(v3[1], ncol);

	for (size_t x = 0; x < nrow; x++) {
		for (size_t y = 0; y < nrow; y++) {
			EXPECT_EQ(ex1[y][x], t1.get({x, y}));
		}
    }
    for (size_t x = 0; x < ncol; x++) {
		for (size_t y = 0; y < ncol; y++) {
			EXPECT_EQ(ex2[y][x], t2.get({x, y}));
			EXPECT_EQ(ex3[y][x], t3.get({x, y}));
		}
    }
}


TEST(OPERATION, matmul2) {
	const size_t limit = 523;
	std::vector<double> av = {
		3, 4,
		41, 6,
		8, 1,
		18, 2,
	};
	std::vector<double> bv = {
		2, 3, 10,
		12, 54, 11,
		0, 9, 0,
		0.1, 29, 0
	};
	std::vector<double> cv = {
		11, 22, 32, 9,
		6, 4, 45, 3.2,
		6, 3, 3, 3,
		12, 10, 22, 32,
		0, 2, 2, 1
	};
	double ex1[2][3] = {
		{499.8, 2817, 481},
		{80.2, 403, 106}
	};
	double ex2[3][5] = {
		{286.9, 60.32, 48.3, 147.2, 24.1},
		{1770, 731.8, 294, 1702, 155},
 		{352, 104, 93, 230, 22}
	};
	double ex3[5][2] = {
		{1353, 226},
		{599.6, 99.4},
		{219, 51},
		{1198, 194},
		{116, 16}
	};
	nnet::placeholder<double> A(std::vector<size_t>{2, 4});
	nnet::placeholder<double> B(std::vector<size_t>{3, 4});
	nnet::placeholder<double> C(std::vector<size_t>{4, 5});
	nnet::matmul<double> ans1(A, B, true); // output is 2x3 (row by col)
	nnet::matmul<double> ans2(B, C, true, true); // output is 3x5
	nnet::matmul<double> ans3(C, A); // output is 5x2

	// didn't initialize
	EXPECT_DEATH({ ans1.eval(); }, ".*");
	EXPECT_DEATH({ ans2.eval(); }, ".*");
	EXPECT_DEATH({ ans3.eval(); }, ".*");
	A = av;
	B = bv;
	C = cv;

	nnet::expose<double> res1(ans1);
	nnet::expose<double> res2(ans2);
	nnet::expose<double> res3(ans3);
	// evaluates
    nnet::tensor<double> t1 = res1.eval();
	nnet::tensor_shape s1 = t1.get_shape();
	std::vector<size_t> v1 = s1.as_list();
	ASSERT_EQ(v1.size(), 2);
	ASSERT_EQ(v1[0], 3);
	ASSERT_EQ(v1[1], 2);
	for (size_t x = 0; x < 3; x++) {
		for (size_t y = 0; y < 2; y++) {
			EXPECT_EQ(ex1[y][x], t1.get({x, y}));
		}
    }

    nnet::tensor<double> t2 = res2.eval();
	nnet::tensor_shape s2 = t2.get_shape();
	std::vector<size_t> v2 = s2.as_list();
	ASSERT_EQ(v2.size(), 2);
	ASSERT_EQ(v2[0], 5);
	ASSERT_EQ(v2[1], 3);
	std::stringstream res;
    for (size_t x = 0; x < 5; x++) {
		for (size_t y = 0; y < 3; y++) {
			EXPECT_EQ(ex2[y][x], t2.get({x, y}));
		}
    }
	std::cout << res.str();

    nnet::tensor<double> t3 = res3.eval();
	nnet::tensor_shape s3 = t3.get_shape();
	std::vector<size_t> v3 = s3.as_list();
	ASSERT_EQ(v3.size(), 2);
	ASSERT_EQ(v3[0], 2);
	ASSERT_EQ(v3[1], 5);
	for (size_t y = 0; y < 5; y++) {
    	for (size_t x = 0; x < 2; x++) {
			EXPECT_EQ(ex3[y][x], t3.get({x, y}));
		}
    }
}


TEST(OPERATION, transpose) {
	std::vector<double> av = {
		3, 4,
		41, 6,
		8, 1,
		18, 2,
	};
	std::vector<double> bv = {
		2, 3, 10,
		12, 54, 11,
		0, 9, 0,
		0.1, 29, 0
	};
	std::vector<double> cv = {
		11, 22, 32, 9,
		6, 4, 45, 3.2,
		6, 3, 3, 3,
		12, 10, 22, 32,
		0, 2, 2, 1
	};
	double ex1[2][4] = {
		{3, 41, 8, 18},
		{4, 6, 1, 2}
	};
	double ex2[3][4] = {
		{2, 12, 0, 0.1},
		{3, 54, 9, 29},
		{10, 11, 0, 0}
	};
	double ex3[4][5] = {
		{11, 6, 6, 12, 0},
		{22, 4, 3, 10, 2},
		{32, 45, 3, 22, 2},
		{9, 3.2, 3, 32, 1}
	};
	nnet::placeholder<double> A(std::vector<size_t>{2, 4});
	nnet::placeholder<double> B(std::vector<size_t>{3, 4});
	nnet::placeholder<double> C(std::vector<size_t>{4, 5});
	nnet::transpose<double> resa(A);
	nnet::transpose<double> resb(B);
	nnet::transpose<double> resc(C);
	A = av;
	B = bv;
	C = cv;
	nnet::tensor<double> ta = resa.eval();
	nnet::tensor_shape sa = ta.get_shape();
	std::vector<size_t> va = sa.as_list();
	ASSERT_EQ(va.size(), 2);
	ASSERT_EQ(va[0], 4);
	ASSERT_EQ(va[1], 2);
    for (size_t x = 0; x < 4; x++) {
		for (size_t y = 0; y < 2; y++) {
			EXPECT_EQ(ex1[y][x], ta.get({x, y}));
		}
	}
	nnet::tensor<double> tb = resb.eval();
	nnet::tensor_shape sb = tb.get_shape();
	std::vector<size_t> vb = sb.as_list();
	ASSERT_EQ(vb.size(), 2);
	ASSERT_EQ(vb[0], 4);
	ASSERT_EQ(vb[1], 3);
    for (size_t x = 0; x < 4; x++) {
		for (size_t y = 0; y < 3; y++) {
			EXPECT_EQ(ex2[y][x], tb.get({x, y}));
		}
	}
	nnet::tensor<double> tc = resc.eval();
	nnet::tensor_shape sc = tc.get_shape();
	std::vector<size_t> vc = sc.as_list();
	ASSERT_EQ(vc.size(), 2);
	ASSERT_EQ(vc[0], 5);
	ASSERT_EQ(vc[1], 4);
    for (size_t x = 0; x < 5; x++) {
		for (size_t y = 0; y < 4; y++) {
			EXPECT_EQ(ex3[y][x], tc.get({x, y}));
		}
	}
}


TEST(OPERATION, dot) {

}


TEST(OPERATION, high_dim_mul) {

}


TEST(OPERATION, contraction) {

}

// DERIVATIVES

TEST(DERIV, unary) {
	const size_t limit = 523;
	const size_t edge = 10;
	const size_t supersize = edge*edge*edge;
	nnet::placeholder<double> in(std::vector<size_t>{edge, edge, edge});
	nnet::placeholder<double> bad(std::vector<size_t>{edge, edge, edge});

	std::vector<nnet::ivariable<double>*> univars = {
		new nnet::neg<double>(in),
		new nnet::sin<double>(in),
		new nnet::cos<double>(in),
		new nnet::tan<double>(in),
		new nnet::csc<double>(in),
		new nnet::sec<double>(in),
		new nnet::cot<double>(in),
		new nnet::exp<double>(in),
	};

	std::vector<std::function<double(double)> > derivs = {
		[](double e) { return -1; },
		[](double e) { return cos(e); },
		[](double e) { return -sin(e); },
		[](double e) { return 1.0/(cos(e)*cos(e)); },
		[](double e) { return -1.0/(sin(e)*tan(e)); },
		[](double e) { return tan(e)/cos(e); },
		[](double e) { return -(1.0/sin(e))*(1.0/sin(e)); },
		[](double e) { return exp(e); },
	};

	// not part of TEST
	assert(univars.size() == derivs.size());

	std::vector<double> r;
	for (size_t i = 0; i < supersize; i++) {
		double given = rand();
		r.push_back(given-round(given/limit));
	}
	in = r;

	for (size_t i = 0; i < univars.size(); i++) {
		nnet::expose<double> exvar(*univars[i]);
		exvar.eval(); // not really needed since we only operation has depth 1
		std::vector<double> badv = exvar.get_derive(bad);
		EXPECT_TRUE(badv.empty());
		std::vector<double> raw = exvar.get_derive(in);
		ASSERT_EQ(supersize, raw.size());
		for (size_t j = 0; j < supersize; j++) {
			EXPECT_EQ(derivs[i](r[j]), raw[j]);
		}
		delete univars[i];
	}
}


TEST(DERIV, binary) {
	const size_t edge = 10;
	const size_t supersize = edge*edge*edge;
	nnet::placeholder<double> a(std::vector<size_t>{edge, edge, edge});
	nnet::placeholder<double> b(std::vector<size_t>{edge, edge, edge});
	nnet::placeholder<double> bad(std::vector<size_t>{edge, edge, edge});

	std::vector<nnet::ivariable<double>*> univars = {
		new nnet::add<double>(a, b),
		new nnet::sub<double>(a, b),
		new nnet::mul<double>(a, b),
		new nnet::div<double>(a, b),
	};

	std::vector<std::function<double(double, double, bool)> > derivs = {
		[](double a, double b, bool is_a) { return 1; },
		[](double a, double b, bool is_a) { return is_a ? 1 : -1; },
		[](double a, double b, bool is_a) { return is_a ? b : a; },
		[](double a, double b, bool is_a) { return is_a ? 1/b : -a/(b*b); },
	};

	// not part of TEST
	assert(univars.size() == derivs.size());

	std::vector<double> ra;
	std::vector<double> rb;
	for (size_t i = 0; i < supersize; i++) {
		ra.push_back(rand());
		rb.push_back(rand());
	}
	a = ra;
	b = rb;

	for (size_t i = 0; i < univars.size(); i++) {
		nnet::expose<double> exvar(*univars[i]);
		exvar.eval(); // not really needed since we only operation has depth 1
		std::vector<double> badv = exvar.get_derive(bad);
		EXPECT_TRUE(badv.empty());

		std::vector<double> rawa = exvar.get_derive(a);
		std::vector<double> rawb = exvar.get_derive(b);
		ASSERT_EQ(supersize, rawa.size());
		ASSERT_EQ(supersize, rawb.size());
		for (size_t j = 0; j < supersize; j++) {
			EXPECT_EQ(derivs[i](ra[j], rb[j], true), rawa[j]);
			EXPECT_EQ(derivs[i](ra[j], rb[j], false), rawb[j]);
		}
		delete univars[i];
	}
}


TEST(DERIV, matop) {

}


TEST(DERIV, complex) {
	const size_t edge = 10;
	const size_t supersize = edge*edge*edge;
	nnet::placeholder<double> p1(std::vector<size_t>{edge, edge, edge});
	nnet::placeholder<double> p2(std::vector<size_t>{edge, edge, edge});

	nnet::sin<double> o1(p1);
	nnet::mul<double> o2(p1, p2);
	nnet::add<double> o3(o1, o2);
	nnet::expose<double> res(o3);

	// pipe into placeholder
	std::vector<double> r1;
	std::vector<double> r2;
	for (size_t i = 0; i < supersize; i++) {
		r1.push_back(rand());
		r2.push_back(rand());
	}
	p1 = r1;
	p2 = r2;

	// res = f(a,b) = sin(a)+a*b
	// df(a,b)/da = cos(a)+b
	// df(a,b)/db = a
	std::vector<double> raw = res.get_raw();
	std::vector<double> dp1 = res.get_derive(p1);
	std::vector<double> dp2 = res.get_derive(p2);
    ASSERT_EQ(raw.size(), supersize);
	ASSERT_EQ(dp1.size(), supersize);
	ASSERT_EQ(dp2.size(), supersize);

    for (size_t i = 0; i < supersize; i++) {
		EXPECT_EQ(sin(r1[i])+r1[i]*r2[i], raw[i]);
		EXPECT_EQ(cos(r1[i])+r2[i], dp1[i]);
		EXPECT_EQ(r1[i], dp2[i]);
    }
}


// TODO test function wrappers
