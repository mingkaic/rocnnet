//
//  test_nnlayer.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <cmath>
#include "gd_net.hpp"
#include "gtest/gtest.h"

#define VECS std::pair<std::vector<double>, std::vector<double> >

static void fill_binary_samples(
	std::vector<VECS>& samples,
	size_t n_input,
	size_t batch_size)
{
	for (size_t i = 0; i < batch_size; i++)
	{
		std::vector<double> a;
		std::vector<double> b;
		for (size_t j = 0; j < n_input/2; j++)
		{
			a.push_back(rand()%2);
		}
		for (size_t j = 0; j < n_input/2; j++)
		{
			b.push_back(rand()%2);
		}
		std::vector<double> c;
		for (size_t j = 0; j < n_input/2; j++)
		{
			c.push_back(((int)a[j])^((int)b[j]));
		}
		a.insert(a.end(), b.begin(), b.end());
		samples.push_back(VECS(a, c));
	}
}


// given a vector of vector pairs,
// concatenate all vectors in the first part of the pair as first
// and all second part as second
static void flatten (const std::vector<VECS>& samples,
	std::vector<double>& first,
	std::vector<double>& second, 
	signed idx = -1)
{
	if (idx < 0)
	{
		for (VECS vp : samples)
		{
			first.insert(first.end(), vp.first.begin(), vp.first.end());
			second.insert(second.end(), vp.second.begin(), vp.second.end());
		}
	}
	else
	{
		first = samples[idx].first;
		second = samples[idx].second;
	}
}


// Each layer perceptron must be reusable by linking its weights to multiple graphs
TEST(PERCEPTRON, layer_multiple_in)
{
	nnet::session& sess = nnet::session::get_instance();
	nnet::placeholder<double> in1((std::vector<size_t>{5}), "layerin1");
	nnet::placeholder<double> in2((std::vector<size_t>{5}), "layerin2");
	// layer connect
	nnet::perceptron layer(5, 5);
	nnet::varptr<double> res1 = layer(&in1);
	nnet::varptr<double> res2 = layer(&in2);
	// initialize weights
	sess.initialize_all<double>();
	// feed in data
	in1 = std::vector<double>{1, 4, 8, 16, 32};
	in2 = std::vector<double>{5, 4, 3, 2, 1};
	// expose
	std::vector<double> raw1 = nnet::expose<double>(res1);
	std::vector<double> raw2 = nnet::expose<double>(res2);
	ASSERT_EQ(raw1.size(), raw2.size());
	EXPECT_FALSE(std::equal(raw1.begin(), raw1.end(), raw2.begin()));
}


TEST(PERCEPTRON, layer_action)
{
	nnet::session& sess = nnet::session::get_instance();
	std::vector<double> vin = {1, 2, 3, 4, 5};
	std::vector<double> exout = {1, 2, 3};
	nnet::perceptron layer(vin.size(), exout.size());
	nnet::placeholder<double> in(std::vector<size_t>{5}, "layerin");
	nnet::varptr<double> res = layer(&in);
	// initialize weight
	sess.initialize_all<double>();
	// feed data
	in = vin;
	std::vector<double> raw = nnet::expose<double>(res);
	ASSERT_EQ(raw.size(), exout.size());
}


TEST(PERCEPTRON, mlp_multiple_in)
{
	nnet::session& sess = nnet::session::get_instance();
	nnet::placeholder<double> in1((std::vector<size_t>{5}), "layerin1");
	nnet::placeholder<double> in2((std::vector<size_t>{5}), "layerin2");
	std::vector<IN_PAIR> hiddens = {
		// use same sigmoid in static memory once deep copy is established
		IN_PAIR(5, nnet::sigmoid<double>),
		IN_PAIR(5, nnet::sigmoid<double>),
		IN_PAIR(5, nnet::sigmoid<double>),
	};
	// mlp create and connect
	nnet::ml_perceptron mlp = nnet::ml_perceptron(5, hiddens);
	nnet::varptr<double> res1 = mlp(&in1);
	nnet::varptr<double> res2 = mlp(&in2);
	// initialize data
	sess.initialize_all<double>();
	// feed data
	in1 = std::vector<double>{1, 4, 8, 16, 32};
	in2 = std::vector<double>{5, 4, 3, 2, 1};
	// expose
	std::vector<double> raw1 = nnet::expose<double>(res1);
	std::vector<double> raw2 = nnet::expose<double>(res2);
	ASSERT_EQ(raw1.size(), raw2.size());
	for (size_t i = 0; i < raw1.size(); i++)
	{
		EXPECT_NE(raw1[i], raw2[i]);
	}
}


TEST(PERCEPTRON, mlp_action)
{
	nnet::session& sess = nnet::session::get_instance();
	size_t n_out = 3;
	size_t n_hidden = 4;
	std::vector<double> vin = {1, 2, 3, 4, 5};
	std::vector<IN_PAIR> hiddens = {
		// use same sigmoid in static memory once deep copy is established
		IN_PAIR(n_hidden, nnet::sigmoid<double>),
		IN_PAIR(n_hidden, nnet::sigmoid<double>),
		IN_PAIR(n_out, nnet::sigmoid<double>),
	};
	nnet::ml_perceptron mlp = nnet::ml_perceptron(vin.size(), hiddens);

	nnet::placeholder<double> in((std::vector<size_t>{5}), "layerin");
	nnet::varptr<double> res = mlp(&in);
	// initialization
	sess.initialize_all<double>();
	// feed data
	in = vin;
	// expose
	std::vector<double> raw = nnet::expose<double>(res);
	ASSERT_EQ(raw.size(), n_out);
	// expect random but non-zero output
	double sum = 0;
	for (double o : raw)
	{
		sum += o;
		EXPECT_LE(o, 1);
		EXPECT_GE(o, 0);
	}
	ASSERT_GT(sum/raw.size(), 0);
}


// relies on optimizer actually working!
// TEST(PERCEPTRON, layer_optimizer)
// {
//  	nnet::gd_optimizer optimizer(0.001);
// 	nnet::session& sess = nnet::session::get_instance();
// 	sess.seed_rand_eng(1);
// 	size_t n_in = 10;
// 	size_t n_out = n_in/2;
// 	size_t n_hidden = 8;
// 	std::vector<IN_PAIR> hiddens = {
// 		// use same sigmoid in static memory once deep copy is established
// 		IN_PAIR(n_hidden, nnet::sigmoid<double>),
// 		IN_PAIR(n_hidden, nnet::sigmoid<double>),
// 		IN_PAIR(n_out, nnet::sigmoid<double>),
// 	};

// 	nnet::ml_perceptron mlp = nnet::ml_perceptron(n_in, hiddens);

// 	nnet::placeholder<double> in((std::vector<size_t>{10}), "layerin");
// 	nnet::varptr<double> res = mlp(&in);
// 	nnet::placeholder<double> expect_out((std::vector<size_t>{5}), "expectout");
// 	nnet::varptr<double> diff = res - nnet::varptr<double>(&expect_out);
// 	nnet::varptr<double> err = diff * diff;

// 	// set up optimizer to minimize error value diff
// 	optimizer.set_root(err);
// 	optimizer.freeze();

// 	// initialize
// 	sess.initialize_all<double>();

// 	// train
// 	size_t train_size = 100;
// 	std::vector<VECS> samples;
// 	for (size_t i = 0; i < train_size; i++)
// 	{
// 		samples.clear();
// 		fill_binary_samples(samples, n_in, 1);
// 		in = samples[0].first;
// 		expect_out = samples[0].second;
// 		// update
// 		optimizer.execute();
// 	}

// 	// test
// 	size_t fails = 0;
// 	size_t test_size = 100;
// 	for (size_t i = 0; i < train_size; i++)
// 	{
// 		samples.clear();
// 		fill_binary_samples(samples, n_in, 1);
// 		in = samples[0].first;
// 		expect_out = samples[0].second;
// 		std::vector<double> err_raw = nnet::expose<double>(err);
// 		// expect error to be very small
// 		for (double e : err_raw)
// 		{
// 			if (0 != std::round(e)) {
// 				fails++;
// 			}
// 		}
// 	}
// 	double successrate = 1.0-(double)fails/(test_size * n_out);
// 	ASSERT_GE(successrate, 0.90); // TODO increase to 90
// 	std::cout << "success rate: " << successrate << std::endl;
// }


// test with adhoc stochastic gradient descent
TEST(PERCEPTRON, gd_train) {
	nnet::session& sess = nnet::session::get_instance();
	sess.seed_rand_eng(1);
	size_t n_in = 10;
	size_t n_out = n_in/2;
	size_t n_hidden = 8;
	std::vector<IN_PAIR> hiddens = {
		// use same sigmoid in static memory once deep copy is established
		IN_PAIR(n_hidden, nnet::sigmoid<double>),
		IN_PAIR(n_hidden, nnet::sigmoid<double>),
		IN_PAIR(n_out, nnet::sigmoid<double>),
	};
	size_t batch_size = 1;
	size_t test_size = 100;
	size_t train_size = 100;
	nnet::gd_net net(n_in, hiddens);
	net.set_the_record_str8(true);
	nnet::placeholder<double> fanin((std::vector<size_t>{n_in, batch_size}), "fanin");
	nnet::varptr<double> fanout = net(&fanin);

	// initialize
	sess.initialize_all<double>();

	std::vector<VECS> samples;
	for (size_t i = 0; i < train_size; i++) {
		std::cout << "training " << i << "\n";
		samples.clear();
		fill_binary_samples(samples, n_in, batch_size);
		net.train(samples[0].first, samples[0].second);
	}

	size_t fails = 0;
	double err = 0;
	for (size_t i = 0; i < test_size; i++) {
		std::cout << "testing " << i << "\n";
		samples.clear();
		fill_binary_samples(samples, n_in, test_size);
		// feed input
		fanin = samples[0].first;
		std::vector<double> res = nnet::expose<double>(fanout);;
		std::vector<double> expect = samples[0].second;
		for (size_t i = 0; i < res.size(); i++) {
			err += std::abs(expect[i] - res[i]);
			if (expect[i] != std::round(res[i])) {
				fails++;
			}
		}
	}
	double successrate = 1.0-(double)fails/(test_size*n_out);
	err /= (test_size*n_out);
	ASSERT_GE(successrate, 0.80); // TODO increase to 90
	std::cout << "average err: " << err << std::endl;
	std::cout << "success rate: " << successrate << std::endl;
}


// // test with gradient descent optimizer
// TEST(PERCEPTRON, gd_optimizer_train) {
// 	nnet::session& sess = nnet::session::get_instance();
// 	sess.seed_rand_eng(1);
// 	size_t n_in = 10;
// 	size_t n_out = n_in/2;
// 	size_t n_hidden = 8;
// 	std::vector<IN_PAIR> hiddens = {
// 		// use same sigmoid in static memory once deep copy is established
// 		IN_PAIR(n_hidden, nnet::sigmoid<double>),
// 		IN_PAIR(n_hidden, nnet::sigmoid<double>),
// 		IN_PAIR(n_out, nnet::sigmoid<double>),
// 	};
// 	size_t batch_size = 1;
// 	size_t test_size = 100;
// 	size_t train_size = 10000;

// 	nnet::OPTIMIZER<double> optimizer = std::make_shared<nnet::gd_optimizer>(0.001);
// 	nnet::gd_net net(n_in, hiddens, optimizer);
// 	nnet::placeptr<double> fanin = new nnet::placeholder<double>((std::vector<size_t>{n_in, batch_size}), "fanin");
// 	nnet::varptr<double> fanout = net(fanin);
// 	nnet::expose<double>* exposeout = new nnet::expose<double>(fanout);
// 	sess.initialize_all<double>();
// 	net.set_the_record_str8(true);

// 	std::vector<VECS> samples;
// 	for (size_t i = 0; i < train_size; i++) {
// 		std::cout << "training " << i << "\n";
// 		samples.clear();
// 		fill_binary_samples(samples, n_in, batch_size);
// 		net.train(samples[0].first, samples[0].second);
// 	}

// 	size_t fails = 0;
// 	double err = 0;
// 	for (size_t i = 0; i < test_size; i++) {
// 		std::cout << "testing " << i << "\n";
// 		samples.clear();
// 		fill_binary_samples(samples, n_in, test_size);

// 		*fanin = samples[0].first;
// 		std::vector<double> res = exposeout->get_raw();
// 		std::vector<double> expect = samples[0].second;
// 		for (size_t i = 0; i < res.size(); i++) {
// 			err += std::abs(expect[i] - res[i]);
// 			if (expect[i] != std::round(res[i])) {
// 				fails++;
// 			}
// 		}
// 	}
// 	double successrate = 1.0-(double)fails/(test_size * n_out);
// 	err /= (test_size * n_out);
// 	ASSERT_GE(successrate, 0.80); // TODO increase to 90
// 	std::cout << "average err: " << err << std::endl;
// 	std::cout << "success rate: " << successrate << std::endl;
// }


// // test with batch gradient descent
// TEST(PERCEPTRON, bgd_train) {
// 	nnet::session& sess = nnet::session::get_instance();
// 	sess.seed_rand_eng(1);
// 	size_t n_in = 10;
// 	size_t n_out = n_in/2;
// 	size_t n_hidden = 8;
// 	std::vector<IN_PAIR> hiddens = {
// 		// use same sigmoid in static memory once deep copy is established
// 		IN_PAIR(n_hidden, nnet::sigmoid<double>),
// 		IN_PAIR(n_hidden, nnet::sigmoid<double>),
// 		IN_PAIR(n_out, nnet::sigmoid<double>),
// 	};
// 	size_t batch_size = 12;
// 	size_t test_size = 100;
// 	size_t train_size = 5000;
// 	nnet::gd_net net(n_in, hiddens);
// 	nnet::placeptr<double> fanin = new nnet::placeholder<double>((std::vector<size_t>{n_in}), "fanin");
// 	nnet::varptr<double> fanout = net(fanin);
// 	nnet::expose<double>* exposeout = new nnet::expose<double>(fanout);
// 	sess.initialize_all<double>();

// 	std::vector<VECS> samples;
// 	for (size_t i = 0; i < train_size; i++) {
// 		std::cout << "training " << i << "\n";
// 		samples.clear();
// 		fill_binary_samples(samples, n_in, batch_size);
// 		std::vector<double> first;
// 		std::vector<double> second;
// 		flatten(samples, first, second);
// 		net.train(first, second);
// 	}

// 	size_t fails = 0;
// 	double err = 0;
// 	for (size_t i = 0; i < test_size; i++) {
// 		std::cout << "testing " << i << "\n";
// 		samples.clear();
// 		fill_binary_samples(samples, n_in, batch_size);
// 		std::vector<double> first;
// 		std::vector<double> second;

// 		flatten(samples, first, second, 0);
// 		*fanin = first;

// 		std::vector<double> res = exposeout->get_raw();
// 		std::vector<double> expect = second;

// 		for (size_t i = 0; i < res.size(); i++) {
// 			err += std::abs(expect[i] - res[i]);
// 			if (expect[i] != std::round(res[i])) {
// 				fails++;
// 			}
// 		}
// 	}
// 	double successrate = 1.0-(double)fails/(test_size * n_out);
// 	err /= (test_size * n_out);
// 	// at least a weak classifier
// 	ASSERT_GE(successrate, 0.55); // TODO: increase to 0.9
// 	std::cout << "average err: " << err << std::endl;
// 	std::cout << "success rate: " << successrate << std::endl;
// }
