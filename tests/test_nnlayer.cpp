//
//  test_nnlayer.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <cmath>
#include "../include/nnet.hpp"
#include "gtest/gtest.h"

static double sigmoid (double in) {
	return 1/(1+exp(-in));
}

static double grad_sig (double in) {
	double sx = sigmoid(in);
	return sx*(1-sx);
}

static double identity (double in) { return in; }

static void fill_binary_samples(
	std::vector<VECS>& samples,
	size_t n_input,
	size_t batch_size) {
	for (size_t i = 0; i < batch_size; i++) {
		std::vector<double> a;
		std::vector<double> b;
		for (size_t j = 0; j < n_input/2; j++) {
			a.push_back(rand()%2);
		}
		for (size_t j = 0; j < n_input/2; j++) {
			b.push_back(rand()%2);
		}
		std::vector<double> c;
		for (size_t j = 0; j < n_input/2; j++) {
			c.push_back(((int)a[j])^((int)b[j]));
		}
		a.insert(a.end(), b.begin(), b.end());
		samples.push_back(VECS(a, c));
	}
}

nnet::adhoc_operation sig(sigmoid, grad_sig);
nnet::adhoc_operation same(identity, identity);


TEST(PERCEPTRON, layer_action) {
	nnet::session& sess = nnet::session::get_instance();
	std::vector<double> vin = {1, 2, 3, 4, 5};
	std::vector<double> exout = {1, 2, 3};
	nnet::layer_perceptron layer(vin.size(), exout.size());
	nnet::placeholder<double> var(std::vector<size_t>{5});
	nnet::ivariable<double>& res = layer(var);
	nnet::expose<double> ex(res);
	// initialize variables
	sess.initialize_all<double>();
	var = vin;
	std::vector<double> raw = ex.get_raw();
	ASSERT_EQ(raw.size(), exout.size());
}


TEST(PERCEPTRON, mlp_action) {
	nnet::session& sess = nnet::session::get_instance();
	size_t n_out = 3;
	size_t n_hidden = 4;
	std::vector<double> vin = {1, 2, 3, 4, 5};
	std::vector<IN_PAIR> hiddens = {
		// use same sigmoid in static memory once deep copy is established
		IN_PAIR(n_hidden, new nnet::sigmoid<double>()),
		IN_PAIR(n_hidden, new nnet::sigmoid<double>()),
		IN_PAIR(n_out, new nnet::sigmoid<double>()),
	};
	// mlp has ownership of activations
	nnet::ml_perceptron mlp =
		nnet::ml_perceptron(vin.size(), hiddens);

	nnet::placeholder<double> var(std::vector<size_t>{5});
	nnet::ivariable<double>& res = mlp(var);
	nnet::expose<double> ex(res);
	sess.initialize_all<double>();
	var = vin;
	std::vector<double> raw = ex.get_raw();
	ASSERT_EQ(raw.size(), n_out);
	for (double o : raw) {
		EXPECT_LE(o, 1);
		EXPECT_GE(o, 0);
	}
}


// test with gradient descent
TEST(PERCEPTRON, gd_train) {
	nnet::session& sess = nnet::session::get_instance();
	sess.seed_rand_eng(1);
	size_t n_in = 10;
	size_t n_out = n_in/2;
	size_t n_hidden = 8;
	std::vector<IN_PAIR> hiddens = {
		// use same sigmoid in static memory once deep copy is established
		IN_PAIR(n_hidden, new nnet::sigmoid<double>()),
		IN_PAIR(n_hidden, new nnet::sigmoid<double>()),
		IN_PAIR(n_out, new nnet::sigmoid<double>()),
	};
	size_t batch_size = 1;
	size_t test_size = 100;
	nnet::gd_net net(n_in, hiddens);
	nnet::placeholder<double> fanin(std::vector<size_t>{n_in, batch_size});
	nnet::placeholder<double> exout(std::vector<size_t>{n_out, batch_size});
	nnet::ivariable<double>& fanout = net(fanin);
	sess.initialize_all<double>();

	std::vector<VECS> samples;
	fill_binary_samples(samples, n_in, batch_size);
	for (VECS vecs : samples) {
		fanin = vecs.first;
		exout = vecs.second;
		//net.train(exout);
	}
}


// test with gradient descent
TEST(PERCEPTRON, gd_train1) {
	size_t n_in = 10;
	size_t n_out = n_in/2;
	std::vector<size_t> n_hidden = {8, 8, n_out};
	size_t batch_size = 50000;
	size_t test_size = 100;

	std::vector<VECS> samples;
	std::vector<std::pair<size_t, nnet::adhoc_operation> > hiddens;

	for (size_t hid_size : n_hidden) {
		hiddens.push_back(std::pair<size_t, nnet::adhoc_operation>(hid_size, sig));
	}

	nnet::gd_net net(n_in, hiddens);
	fill_binary_samples(samples, n_in, batch_size);
	for (VECS s : samples) {
		net.train(s);
	}

	samples.clear();
	fill_binary_samples(samples, n_in, test_size);

	size_t fails = 0;
	double err = 0;
	for (VECS s : samples) {
		std::vector<double> out = net(s.first);
		for (size_t i = 0; i < n_out; i++) {
			err += std::abs(s.second[i]-out[i]);
			if (s.second[i] != round(out[i])) {
				fails++;
			}
		}
	}
	double successrate = 1.0-(double)fails/(samples.size()*n_in);
	err /= (test_size*n_out);
	ASSERT_GE(successrate, 0.9);
	std::cout << "average err: " << err << std::endl;
	std::cout << "success rate: " << successrate << std::endl;
}


// test with batch gradient descent
TEST(PERCEPTRON, bgd_train) {
	size_t n_in = 10;
	size_t n_out = n_in/2;
	std::vector<size_t> n_hidden = {10, n_out};
	size_t trials = 100;
	size_t batch_size = 20;
	size_t test_size = 20000;

	std::vector<VECS> samples;
	std::vector<std::pair<size_t, nnet::adhoc_operation> > hiddens;

	for (size_t hid_size : n_hidden) {
		hiddens.push_back(std::pair<size_t, nnet::adhoc_operation>(hid_size, sig));
	}

	nnet::gd_net net(n_in, hiddens);

	for (size_t i = 0; i < trials; i++) {
		samples.clear();
		fill_binary_samples(samples, n_in, batch_size);
		net.train(samples);
	}

	samples.clear();
	fill_binary_samples(samples, n_in, test_size);

	size_t fails = 0;
	double err = 0;
	for (VECS s : samples) {
		std::vector<double> out = net(s.first);
		for (size_t i = 0; i < n_out; i++) {
			err += std::abs(s.second[i]-out[i]);
			if (s.second[i] != round(out[i])) {
				fails++;
			}
		}
	}
	double successrate = 1.0-(double)fails/(samples.size()*n_in);
	err /= (test_size*n_out);
	ASSERT_GE(successrate, 0.75);
	std::cout << "average err: " << err << std::endl;
	std::cout << "success rate: " << successrate << std::endl;
}
