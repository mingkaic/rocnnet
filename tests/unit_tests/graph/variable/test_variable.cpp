//
// Created by Mingkai Chen on 2016-08-29.
//

#include "memory/session.hpp"
#include "graph/variable/variable.hpp"

TEST(VARIABLE, const_init) {
	nnet::session& sess = nnet::session::get_instance();
	const double constant = (double) rand();
	nnet::const_init<double> init(constant);

	nnet::varptr<double> cvar = new nnet::variable<double>(std::vector<size_t>{3, 3}, init, "constant_arr");

	sess.initialize_all<double>();
	expose<double>* ex = new nnet::expose<double>(cvar);
	std::vector<double> raw = ex->get_raw();

	ASSERT_EQ(raw.size(), 9);

	for (double elem : raw) {
		EXPECT_EQ(elem, constant);
	}
}


TEST(VARIABLE, get_index) {
	nnet::session& sess = nnet::session::get_instance();
	const double constant = (double) rand();
	nnet::const_init<double> init(constant);
	size_t nrows = 3;
	size_t ncols = 4;
	size_t ndeps = 5; // depth

	nnet::varptr<double> cvar =
		new nnet::variable<double>(std::vector<size_t>{nrows, ncols, ndeps}, init, "constant_arr");

	sess.initialize_all<double>();
	nnet::expose<double>* ex = new nnet::expose<double>(cvar);
	nnet::tensor<double> raw = ex->eval();

	ASSERT_EQ(raw.n_elems(), nrows*ncols*ndeps);

	for (size_t i = 0; i < nrows; i++) {
		for (size_t j = 0; j < ncols; j++) {
			for (size_t k = 0; k < ndeps; k++) {
				EXPECT_EQ(constant, raw.get({i,j,k}));
			}
		}
	}
}


TEST(VARIABLE, random_init) {
	nnet::session& sess = nnet::session::get_instance();
	nnet::random_uniform<double> init(0.0, 1.0);

	nnet::varptr<double> rvar = new nnet::variable<double>(std::vector<size_t>{5, 5}, init, "random_arr");

	sess.initialize_all<double>();
	nnet::expose<double>* ex = new nnet::expose<double>(rvar);
	std::vector<double> raw = ex->get_raw();

	ASSERT_EQ(raw.size(), 25);

	double sum = 0;
	for (double elem : raw) {
		sum += elem;
		EXPECT_GT(elem, 0);
		EXPECT_LT(elem, 1);
	}
	ASSERT_GT(sum/raw.size(), 0);
}


TEST(VARIABLE, placeholder) {
	const size_t insize = 20;
	nnet::placeptr<double> invar = new nnet::placeholder<double>((std::vector<size_t>{1, insize}), "in");
	std::vector<double> sample;
	for (size_t i = 0; i < insize; i++) {
		sample.push_back(rand());
	}
	*invar = sample;

	expose<double>* ex = new nnet::expose<double>(invar);
	std::vector<double> raw = ex->get_raw();

	ASSERT_EQ(raw.size(), insize);

	for (size_t i = 0; i < insize; i++) {
		EXPECT_EQ(sample[i], raw[i]);
	}
}
