//
//  test_operation.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "../shared/utils.hpp"

void unaryElemTest (
	std::function<nnet::VAR_PTR<double>(nnet::VAR_PTR<double>)> func,
	std::function<double(double)> op) {
	const size_t limit = 523;
	const size_t edge = 10;
	const size_t supersize = edge*edge*edge;
	nnet::PLACEHOLDER_PTR<double> p = nnet::placeholder<double>::make(std::vector<size_t>{edge, edge, edge}, "unar_in");

	nnet::VAR_PTR<double> res =
		func(std::static_pointer_cast<nnet::ivariable<double>, nnet::placeholder<double> >(p));

	// didn't initialize
	EXPECT_DEATH({ res->eval(); }, ".*");

	std::vector<double> r;
	for (size_t i = 0; i < supersize; i++) {
		double val = 0;
		while (0 == val) {
			val = fmod(rand(), limit);
		}
		r.push_back(val);
	}
	*p = r;

	EXPOSE_PTR ex = nnet::expose<double>::make(res);
	// evaluates
    std::vector<double> raw = ex->get_raw();

    ASSERT_EQ(raw.size(), supersize);

    for (size_t i = 0; i < supersize; i++) {
    	double err = std::abs(op(r[i]) - raw[i]);
		EXPECT_LT(err, 0.001);
    }
}

void binaryElemTest (
	std::function<nnet::VAR_PTR<double>(nnet::VAR_PTR<double>, nnet::VAR_PTR<double>)> func,
	std::function<nnet::VAR_PTR<double>(nnet::VAR_PTR<double>, double)> func1,
	std::function<nnet::VAR_PTR<double>(double, nnet::VAR_PTR<double>)> func2,
	std::function<double(double, double)> op) {
	nnet::session& sess = nnet::session::get_instance();
	sess.enable_shape_eval();
	const size_t limit = 523;
	const size_t edge = 10;
	const size_t badsize = edge*edge;
	const size_t supersize = edge*edge*edge;
	nnet::tensor_shape goodshape = std::vector<size_t>{edge, edge, edge};
	nnet::PLACEHOLDER_PTR<double> p1 = nnet::placeholder<double>::make(goodshape, "bin_in1");
	nnet::PLACEHOLDER_PTR<double> p2 = nnet::placeholder<double>::make(goodshape, "bin_in2");
	nnet::PLACEHOLDER_PTR<double> bad = nnet::placeholder<double>::make((std::vector<size_t>{edge, edge}), "bad");

	EXPECT_DEATH({ nnet::VAR_PTR<double> trouble1 = func(p1, bad); }, ".*"); // evaluates shapes at construction
    EXPECT_DEATH({ nnet::VAR_PTR<double> trouble2 = func(bad, p1); }, ".*");

    nnet::VAR_PTR<double> res = func(p1, p2);
	nnet::VAR_PTR<double> res1 = func1(p1, 2);
	nnet::VAR_PTR<double> _1res = func2(2, p1);

	// didn't initialize
	EXPECT_DEATH({ res->eval(); }, ".*");
	EXPECT_DEATH({ res1->eval(); }, ".*");
	EXPECT_DEATH({ _1res->eval(); }, ".*");

	// initialize
	std::vector<double> r1;
	std::vector<double> r2;
	std::vector<double> rbad;
	for (size_t i = 0; i < supersize; i++) {
		double val = 0;
		while (0 == val) {
			val = fmod(rand(), limit);
		}
		r1.push_back(val);
		val = 0;
		while (0 == val) {
			val = fmod(rand(), limit);
		}
		r2.push_back(val);
	}
	for (size_t i = 0; i < badsize; i++) {
		rbad.push_back(rand());
	}
	*p1 = r1;
	*p2 = r2;

	EXPOSE_PTR ex = nnet::expose<double>::make(res);
	EXPOSE_PTR ex1 = nnet::expose<double>::make(res1);
	EXPOSE_PTR ex2 = nnet::expose<double>::make(_1res);
	// evaluates
    std::vector<double> raw = ex->get_raw();
	std::vector<double> raw1 = ex1->get_raw();
	std::vector<double> raw2 = ex2->get_raw();

    ASSERT_EQ(raw.size(), supersize);
	ASSERT_EQ(raw1.size(), supersize);
	ASSERT_EQ(raw2.size(), supersize);

    for (size_t i = 0; i < supersize; i++) {
		EXPECT_EQ(op(r1[i], r2[i]), raw[i]);
		EXPECT_EQ(op(r1[i], 2), raw1[i]);
		EXPECT_EQ(op(2, r1[i]), raw2[i]);
    }
	sess.disable_shape_eval();
}


TEST(OPERATION, abs) {
	unaryElemTest([](nnet::VAR_PTR<double> in) { return +in; },
	[](double var) { return +var; });
}


TEST(OPERATION, neg) {
	unaryElemTest([](nnet::VAR_PTR<double> in) { return -in; },
	[](double var) { return -var; });
}


TEST(OPERATION, sin) {
	unaryElemTest([](nnet::VAR_PTR<double> in) { return nnet::sin(in); },
	[](double var) { return sin(var); });
}


TEST(OPERATION, cos) {
	unaryElemTest([](nnet::VAR_PTR<double> in) { return nnet::cos(in); },
	[](double var) { return cos(var); });

}


TEST(OPERATION, tan) {
	unaryElemTest([](nnet::VAR_PTR<double> in) { return nnet::tan(in); },
	[](double var) { return tan(var); });
}


TEST(OPERATION, csc) {
	unaryElemTest([](nnet::VAR_PTR<double> in) { return nnet::csc(in); },
	[](double var) { return 1/sin(var); });
}


TEST(OPERATION, sec) {
	unaryElemTest([](nnet::VAR_PTR<double> in) { return nnet::sec(in); },
	[](double var) { return 1/cos(var); });
}


TEST(OPERATION, cot) {
	unaryElemTest([](nnet::VAR_PTR<double> in) { return nnet::cot(in); },
	[](double var) { return cos(var)/sin(var); });
}


TEST(OPERATION, exp) {
	unaryElemTest([](nnet::VAR_PTR<double> in) { return nnet::exp(in); },
	[](double var) { return exp(var); });
}


TEST(OPERATION, add) {
	binaryElemTest(
	[](nnet::VAR_PTR<double> a, nnet::VAR_PTR<double> b) { return a+b; },
	[](nnet::VAR_PTR<double> a, double b) { return a+b; },
	[](double a, nnet::VAR_PTR<double> b) { return a+b; },
	[](double a, double b) { return a+b; });
}


TEST(OPERATION, sub) {
	binaryElemTest(
	[](nnet::VAR_PTR<double> a, nnet::VAR_PTR<double> b) { return a-b; },
	[](nnet::VAR_PTR<double> a, double b) { return a-b; },
	[](double a, nnet::VAR_PTR<double> b) { return a-b; },
	[](double a, double b) { return a-b; });
}


TEST(OPERATION, mul) {
	binaryElemTest(
	[](nnet::VAR_PTR<double> a, nnet::VAR_PTR<double> b) { return a*b; },
	[](nnet::VAR_PTR<double> a, double b) { return a*b; },
	[](double a, nnet::VAR_PTR<double> b) { return a*b; },
	[](double a, double b) { return a*b; });
}


TEST(OPERATION, div) {
	binaryElemTest(
	[](nnet::VAR_PTR<double> a, nnet::VAR_PTR<double> b) { return a/b; },
	[](nnet::VAR_PTR<double> a, double b) { return a/b; },
	[](double a, nnet::VAR_PTR<double> b) { return a/b; },
	[](double a, double b) { return a/b; });
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
	nnet::PLACEHOLDER_PTR<double> A = nnet::placeholder<double>::make((std::vector<size_t>{ncol, nrow}), "a");
	nnet::PLACEHOLDER_PTR<double> B = nnet::placeholder<double>::make((std::vector<size_t>{ncol, nrow}), "b");
	nnet::PLACEHOLDER_PTR<double> C = nnet::placeholder<double>::make((std::vector<size_t>{nrow, ncol}), "c");
	nnet::VAR_PTR<double> ans1 = nnet::matmul<double>::make(A, B, false, true); // output is 4x4
	nnet::VAR_PTR<double> ans2 = nnet::matmul<double>::make(A, B, true); // output is 3x3
	nnet::VAR_PTR<double> ans3 = nnet::matmul<double>::make(C, A); // output is 3x3

	// didn't initialize
	EXPECT_DEATH({ ans1->eval(); }, ".*");
	EXPECT_DEATH({ ans2->eval(); }, ".*");
	EXPECT_DEATH({ ans3->eval(); }, ".*");
	*A = av;
	*B = bv;
	*C = cv;

	EXPOSE_PTR res1 = nnet::expose<double>::make(ans1);
	EXPOSE_PTR res2 = nnet::expose<double>::make(ans2);
	EXPOSE_PTR res3 = nnet::expose<double>::make(ans3);
	// evaluates
    nnet::tensor<double> t1 = res1->eval();
	nnet::tensor_shape s1 = t1.get_shape();
	std::vector<size_t> v1 = s1.as_list();
	ASSERT_EQ(v1.size(), 2);
	ASSERT_EQ(v1[0], nrow);
	ASSERT_EQ(v1[1], nrow);

    nnet::tensor<double> t2 = res2->eval();
	nnet::tensor_shape s2 = t2.get_shape();
	std::vector<size_t> v2 = s2.as_list();
	ASSERT_EQ(v2.size(), 2);
	ASSERT_EQ(v2[0], ncol);
	ASSERT_EQ(v2[1], ncol);

    nnet::tensor<double> t3 = res3->eval();
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
	nnet::PLACEHOLDER_PTR<double> A = nnet::placeholder<double>::make((std::vector<size_t>{2, 4}), "a");
	nnet::PLACEHOLDER_PTR<double> B = nnet::placeholder<double>::make((std::vector<size_t>{3, 4}), "b");
	nnet::PLACEHOLDER_PTR<double> C = nnet::placeholder<double>::make((std::vector<size_t>{4, 5}), "c");
	nnet::VAR_PTR<double> ans1 = nnet::matmul<double>::make(A, B, true); // output is 2x3 (row by col)
	nnet::VAR_PTR<double> ans2 = nnet::matmul<double>::make(B, C, true, true); // output is 3x5
	nnet::VAR_PTR<double> ans3 = nnet::matmul<double>::make(C, A); // output is 5x2

	// didn't initialize
	EXPECT_DEATH({ ans1->eval(); }, ".*");
	EXPECT_DEATH({ ans2->eval(); }, ".*");
	EXPECT_DEATH({ ans3->eval(); }, ".*");
	*A = av;
	*B = bv;
	*C = cv;

	EXPOSE_PTR res1 = nnet::expose<double>::make(ans1);
	EXPOSE_PTR res2 = nnet::expose<double>::make(ans2);
	EXPOSE_PTR res3 = nnet::expose<double>::make(ans3);
	// evaluates
    nnet::tensor<double> t1 = res1->eval();
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

    nnet::tensor<double> t2 = res2->eval();
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

    nnet::tensor<double> t3 = res3->eval();
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
	nnet::PLACEHOLDER_PTR<double> A = nnet::placeholder<double>::make((std::vector<size_t>{2, 4}), "a");
	nnet::PLACEHOLDER_PTR<double> B = nnet::placeholder<double>::make((std::vector<size_t>{3, 4}), "b");
	nnet::PLACEHOLDER_PTR<double> C = nnet::placeholder<double>::make((std::vector<size_t>{4, 5}), "c");
	nnet::VAR_PTR<double> resa = nnet::transpose<double>::make(A);
	nnet::VAR_PTR<double> resb = nnet::transpose<double>::make(B);
	nnet::VAR_PTR<double> resc = nnet::transpose<double>::make(C);
	*A = av;
	*B = bv;
	*C = cv;
	nnet::tensor<double> ta = resa->eval();
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
	nnet::tensor<double> tb = resb->eval();
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
	nnet::tensor<double> tc = resc->eval();
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


TEST(OPERATION, extend) {
	nnet::PLACEHOLDER_PTR<double> A = nnet::placeholder<double>::make((std::vector<size_t>{2, 1, 2}), "a");
	nnet::PLACEHOLDER_PTR<double> B = nnet::placeholder<double>::make((std::vector<size_t>{2, 2, 1}), "b");
	nnet::PLACEHOLDER_PTR<double> C = nnet::placeholder<double>::make((std::vector<size_t>{2, 2, 2}), "c");

	std::vector<double> t1 = {
		0.4, 0.9,
		1.2, 3.1,
	};
	std::vector<double> t2 = {
		// layer 1
		0.4, 0.9,
		1.2, 3.1,
		// layer 2
		1.9, 1.0,
		2.5, 2.0,
	};
	std::vector<double> ex1 = {
		// layer 1
		0.4, 0.9, 0.4, 0.9,
		1.2, 3.1, 1.2, 3.1,
		// layer 2
		1.9, 1.0, 1.9, 1.0,
		2.5, 2.0, 2.5, 2.0,
	};
	std::vector<double> ex2 = {
		// layer 1
		0.4, 0.9,
		0.4, 0.9,
		// layer 2
		1.2, 3.1,
		1.2, 3.1,
	};
	std::vector<double> ex3 = {
		// layer 1
		0.4, 0.9,
		1.2, 3.1,
		// layer 2
		0.4, 0.9,
		1.2, 3.1,
	};
	std::vector<double> ex4 = {
		// layer A
		// layer 1
		0.4, 0.9,
		1.2, 3.1,
		// layer 2
		1.9, 1.0,
		2.5, 2.0,
		// layer B
		// layer 1
		0.4, 0.9,
		1.2, 3.1,
		// layer 2
		1.9, 1.0,
		2.5, 2.0,
	};
	*A = t1;
	*B = t1;
	*C = t2;

	nnet::VAR_PTR<double> e1 = nnet::extend<double>::make(C, 0, 2);
	nnet::VAR_PTR<double> e2 = nnet::extend<double>::make(A, 1, 2);
	nnet::VAR_PTR<double> e3 = nnet::extend<double>::make(B, 2, 2);
	nnet::VAR_PTR<double> e4 = nnet::extend<double>::make(C, 3, 2);

	EXPOSE_PTR expose1 = nnet::expose<double>::make(e1);
	const nnet::tensor<double>& res1 = expose1->eval();
	std::vector<double> raw = expose1->get_raw();

	std::vector<size_t> ts1 = res1.get_shape().as_list();
	ASSERT_EQ(3, ts1.size());
	ASSERT_EQ(4, ts1[0]);
	for (auto it = ++ts1.begin(); it != ts1.end(); it++) {
		ASSERT_EQ(2, *it);
	}
	for (size_t i = 0; i < raw.size(); i++) {
		EXPECT_EQ(ex1[i], raw[i]);
	}

	EXPOSE_PTR expose2 = nnet::expose<double>::make(e2);
	const nnet::tensor<double>& res2 = expose2->eval();
	raw = expose2->get_raw();

	std::vector<size_t> ts2 = res2.get_shape().as_list();
	ASSERT_EQ(3, ts2.size());
	for (size_t s : ts2) {
		ASSERT_EQ(2, s);
	}
	for (size_t i = 0; i < raw.size(); i++) {
		EXPECT_EQ(ex2[i], raw[i]);
	}

	EXPOSE_PTR expose3 = nnet::expose<double>::make(e3);
	const nnet::tensor<double>& res3 = expose3->eval();
	raw = expose3->get_raw();

	std::vector<size_t> ts3 = res3.get_shape().as_list();
	ASSERT_EQ(3, ts3.size());
	for (size_t s : ts3) {
		ASSERT_EQ(2, s);
	}
	for (size_t i = 0; i < raw.size(); i++) {
		EXPECT_EQ(ex3[i], raw[i]);
	}

	EXPOSE_PTR expose4 = nnet::expose<double>::make(e4);
	const nnet::tensor<double>& res4 = expose4->eval();
	raw = expose4->get_raw();

	std::vector<size_t> ts4 = res4.get_shape().as_list();
	ASSERT_EQ(4, ts4.size());
	for (size_t s : ts4) {
		ASSERT_EQ(2, s);
	}
	for (size_t i = 0; i < raw.size(); i++) {
		EXPECT_EQ(ex4[i], raw[i]);
	}

}


TEST(OPERATION, compress) {
	nnet::PLACEHOLDER_PTR<double> A = nnet::placeholder<double>::make((std::vector<size_t>{5, 2}), "a");
	nnet::PLACEHOLDER_PTR<double> B = nnet::placeholder<double>::make((std::vector<size_t>{2, 5}), "b");
	nnet::PLACEHOLDER_PTR<double> C = nnet::placeholder<double>::make((std::vector<size_t>{2, 5, 2}), "c");

	std::vector<double> in1 = {
		1, 2, 3, 4, 5,
		2, 10, 23, 1, 2,
	};

	std::vector<double> exp1 = { 3, 7.6 };

	std::vector<double> in2 = {
		3, 2,
		10, 23,
		2, 1.2,
		0.5, 0.1,
		1, 2,
	};

	std::vector<double> exp2 = { 3.3, 5.66 };

	std::vector<double> in3 = {
		// layer 1
		3, 2,
		10, 23,
		2, 1.2,
		0.5, 0.1,
		1, 2,
		// layer 2
		2, 1.8,
		12, 84,
		92, 1.9,
		9, 3.14,
		70, 17.1,
	};

	std::vector<double> exp3 = {
		// layer 1
		3.3, 5.66,
		// layer 2
		37, 21.588,
	};

	*A = in1;
	*B = in2;
	*C = in3;

	nnet::VAR_PTR<double> c1 = nnet::compress<double>::make(A, 0); // expect vector of 2
	nnet::VAR_PTR<double> c2 = nnet::compress<double>::make(B, 1); // expect vector of 2
	nnet::VAR_PTR<double> c3 = nnet::compress<double>::make(C, 1); // expect shape of 2, 1, 2

	EXPOSE_PTR e1 = nnet::expose<double>::make(c1);
	std::vector<size_t> v1 = e1->eval().get_shape().as_list();
	ASSERT_EQ(1, v1.size());
	ASSERT_EQ(2, v1[0]);
	std::vector<double> raw = e1->get_raw();

	for (size_t i = 0; i < raw.size(); i++) {
		EXPECT_EQ(exp1[i], raw[i]);
	}

	EXPOSE_PTR e2 = nnet::expose<double>::make(c2);
	std::vector<size_t> v2 = e2->eval().get_shape().as_list();
	ASSERT_EQ(1, v2.size());
	ASSERT_EQ(2, v2[0]);
	raw = e2->get_raw();

	for (size_t i = 0; i < raw.size(); i++) {
		EXPECT_EQ(exp2[i], raw[i]);
	}

	EXPOSE_PTR e3 = nnet::expose<double>::make(c3);
	std::vector<size_t> v3 = e3->eval().get_shape().as_list();
	ASSERT_EQ(3, v3.size());
	ASSERT_EQ(2, v3[0]);
	ASSERT_EQ(1, v3[1]);
	ASSERT_EQ(2, v3[2]);
	raw = e3->get_raw();

	for (size_t i = 0; i < raw.size(); i++) {
		EXPECT_EQ(exp3[i], raw[i]);
	}
}


TEST(OPERATION, dot) {

}


TEST(OPERATION, high_dim_mul) {

}

// DERIVATIVES

TEST(DERIV, unary) {
	const size_t edge = 10;
	const size_t supersize = edge*edge*edge;
	nnet::session& sess = nnet::session::get_instance();
	nnet::random_uniform<double> rinit(0, 523);
	nnet::VAR_PTR<double> in = nnet::variable<double>::make((std::vector<size_t>{edge, edge, edge}), rinit, "in");
	//nnet::VAR_PTR<double> bad = MAKE_VARIABLE((std::vector<size_t>{edge, edge, edge}), "bad");

	std::vector<nnet::VAR_PTR<double> > univars = {
		+in,
		-in,
		nnet::sin(in),
		nnet::cos(in),
		nnet::tan(in),
		nnet::csc(in),
		nnet::sec(in),
		nnet::cot(in),
		nnet::exp(in),
	};

	std::vector<std::function<double(double)> > derivs = {
		[](double e) { return +1; },
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

	sess.initialize_all<double>();
	std::vector<double> expect_out = nnet::expose<double>::make(in)->get_raw();
	size_t len = univars.size();

	for (size_t i = 0; i < len; i++) {
		EXPOSE_PTR ex = nnet::expose<double>::make(univars[i]);
		std::vector<double> raw = ex->get_derive(in);
		for (size_t j = 0; j < raw.size(); j++) {
			EXPECT_EQ(derivs[i](expect_out[j]), raw[j]);
		}
	}
}


TEST(DERIV, binary) {
	const size_t edge = 10;
	const size_t supersize = edge*edge*edge;
	nnet::session& sess = nnet::session::get_instance();
	nnet::random_uniform<double> rinit(0, 523);
	nnet::VAR_PTR<double> a = nnet::variable<double>::make((std::vector<size_t>{edge, edge, edge}), rinit, "a");
	nnet::VAR_PTR<double> b = nnet::variable<double>::make((std::vector<size_t>{edge, edge, edge}), rinit, "b");
	nnet::VAR_PTR<double> bad = nnet::variable<double>::make((std::vector<size_t>{edge, edge, edge}), rinit, "bad");

	std::vector<nnet::VAR_PTR<double> > univars = {
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

	std::vector<double> ra = nnet::expose<double>::make(a)->get_raw();
	std::vector<double> rb = nnet::expose<double>::make(b)->get_raw();

	for (size_t i = 0; i < univars.size(); i++) {
		EXPOSE_PTR exvar = nnet::expose<double>::make(univars[i]);
		std::vector<double> badv = exvar->get_derive(bad);
		EXPECT_EQ(0, badv[0]);
		std::vector<double> rawa = exvar->get_derive(a);
		std::vector<double> rawb = exvar->get_derive(b);
		ASSERT_EQ(rawa.size(), rawb.size());
		for (size_t j = 0; j < rawa.size(); j++) {
			double erra = derivs[i](ra[j], rb[j], true) - rawa[j];
			double errb = derivs[i](ra[j], rb[j], false) - rawb[j];
			EXPECT_LT(erra, 0.001);
			EXPECT_LT(errb, 0.001);
		}
	}
}


// TODO: write these
TEST(DERIV, transpose) {

}


TEST(DERIV, matop) {

}


TEST(DERIV, complex) {
	const size_t edge = 10;
	const size_t supersize = edge*edge*edge;
	nnet::session& sess = nnet::session::get_instance();
	nnet::random_uniform<double> rinit(0, 523);
	nnet::VAR_PTR<double> p1 = nnet::variable<double>::make((std::vector<size_t>{edge, edge, edge}), rinit, "p1");
	nnet::VAR_PTR<double> p2 = nnet::variable<double>::make((std::vector<size_t>{edge, edge, edge}), rinit, "p2");

	sess.initialize_all<double>();
	nnet::VAR_PTR<double> o = nnet::sin(p1) + p1 * p2;
	EXPOSE_PTR res = nnet::expose<double>::make(o);

	std::vector<double> r1 = nnet::expose<double>::make(p1)->get_raw();
	std::vector<double> r2 = nnet::expose<double>::make(p2)->get_raw();

	// res = f(a,b) = sin(a)+a*b
	// df(a,b)/da = cos(a)+b
	// df(a,b)/db = a
	std::vector<double> raw = res->get_raw();
	std::vector<double> dp1 = res->get_derive(p1);
	std::vector<double> dp2 = res->get_derive(p2);

    for (size_t i = 0; i < raw.size(); i++) {
		EXPECT_EQ(sin(r1[i])+r1[i]*r2[i], raw[i]);
    }
    for (size_t i = 0; i < dp1.size(); i++) {
		EXPECT_EQ(cos(r1[i])+r2[i], dp1[i]);
	}
	for (size_t i = 0; i < dp2.size(); i++) {
		EXPECT_EQ(r1[i], dp2[i]);
	}
}


// tests deriving with respect to leaf (variable) nodes using sigmoid
TEST(DERIV, sigmoid_complex) {
	const size_t edge = 10;
	const size_t supersize = edge*edge*edge;
	nnet::session& sess = nnet::session::get_instance();
	nnet::random_uniform<double> rinit(0, 523);
	nnet::VAR_PTR<double> x = nnet::variable<double>::make((std::vector<size_t>{edge, edge, edge}), rinit, "p1");

	sess.initialize_all<double>();
	nnet::VAR_PTR<double> o = nnet::sigmoid(x);
	EXPOSE_PTR res = nnet::expose<double>::make(o);

	std::vector<double> xin = nnet::expose<double>::make(x)->get_raw();

	std::function<double(double)> sig = [](double x) {
		return 1 / (1 + std::exp(-x));
	};
	std::function<double(double)> sig_prime = [&sig](double x) {
		double s = sig(x);
		return s * (1 - s);
	};

	std::vector<double> raw = res->get_raw();
	std::vector<double> der = res->get_derive(x);

	for (size_t i = 0; i < raw.size(); i++) {
		EXPECT_EQ(sig(xin[i]), raw[i]);
	}
	for (size_t i = 0; i < der.size(); i++) {
		double err = std::abs(sig_prime(xin[i]) - der[i]);
		EXPECT_LT(err, 0.0001);
	}
}


// tests deriving with respect to operation nodes
TEST(DERIV, operation_derive) {
	const size_t limit = 523;
	const size_t edge = 10;
	const size_t supersize = edge*edge;
	nnet::session& sess = nnet::session::get_instance();
	nnet::random_uniform<double> rinit(0, 523);

	nnet::VAR_PTR<double> x = nnet::variable<double>::make((std::vector<size_t>{edge, edge}), rinit, "p1");
	nnet::PLACEHOLDER_PTR<double> place = nnet::placeholder<double>::make(std::vector<size_t>{edge, edge}, "in");
	nnet::VAR_PTR<double> mul = nnet::matmul<double>::make(x, place); // <X, IN>

	nnet::VAR_PTR<double> o = nnet::sigmoid(mul); // 1/(1+e^(-<X, IN>))
	nnet::VAR_PTR<double> better_grad = o*(1.0-o); // d(1/(1+e^(-<X, IN>))) / d(<X, IN>)
	nnet::VAR_PTR<double> grad = nnet::derive<double>::make(o, mul); // d(1/(1+e^(-<X, IN>))) / d(<X, IN>)
	EXPOSE_PTR oex = nnet::expose<double>::make(o);
	EXPOSE_PTR ex = nnet::expose<double>::make(grad);
	EXPOSE_PTR better_ex = nnet::expose<double>::make(better_grad);
	sess.initialize_all<double>();

	std::vector<double> placeholder_in;
	for (size_t i = 0; i < supersize; i++) {
		placeholder_in.push_back(fmod(rand(), limit));
	}
	*place = placeholder_in;

	std::vector<double> xin = nnet::expose<double>::make(mul)->get_raw();
	std::function<double(double)> sig = [](double x) {
		return 1 / (1 + std::exp(-x));
	};
	std::function<double(double)> sig_prime = [&sig](double x) {
		double s = sig(x);
		return s * (1 - s);
	};

	std::vector<double> der = ex->get_raw();
	std::vector<double> raw = better_ex->get_raw();
	std::vector<double> der_raw = oex->get_derive(mul);

	ASSERT_EQ(der.size(), xin.size());
	ASSERT_EQ(der.size(), raw.size());
	ASSERT_EQ(der.size(), der_raw.size());

	for (size_t i = 0; i < der.size(); i++) {
		// allow some errors since sig_prime and better ex are prone to rounding errors
		// sig_prime is a 2 step process:
		// 1. get sig which can cause rounding
		// 2. taken sig * (1 - sig) which can cause further rounding at 1 - sig
		EXPECT_LT(std::abs(sig_prime(xin[i]) - der[i]), 0.0001);
		EXPECT_LT(std::abs(raw[i] - der[i]), 0.0001);

		// expect gradient buffer eval and expose get gradient to return the EXACT same thing
		EXPECT_EQ(der_raw[i], der[i]);
	}
}


TEST(OPERATION, univar_func) {
	nnet::PLACEHOLDER_PTR<double> fanin = nnet::placeholder<double>::make((std::vector<size_t>{1}), "in");
	nnet::VAR_PTR<double> res = nnet::sigmoid<double>(fanin);
	EXPOSE_PTR out = nnet::expose<double>::make(res);

	*fanin = std::vector<double>{0};
	double sigres = out->get_raw()[0];
	ASSERT_EQ(sigres, 0.5);

	*fanin = std::vector<double>{1};
	sigres = out->get_raw()[0];
	double err = 0.73105857863 - sigres;
	ASSERT_LT(err, 0.0001);
}