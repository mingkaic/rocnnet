//
//  test_nnlayer.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <cmath>
#include "rocnnet/nnet.hpp"
#include "../shared/utils.hpp"

#define VECS std::pair<std::vector<double>, std::vector<double> >

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

static void flatten (const std::vector<VECS>& samples,
	std::vector<double>& first,
	std::vector<double>& second, signed idx = -1) {
	if (idx < 0) {
		for (VECS vp : samples) {
			first.insert(first.end(), vp.first.begin(), vp.first.end());
			second.insert(second.end(), vp.second.begin(), vp.second.end());
		}
	} else {
		first = samples[idx].first;
		second = samples[idx].second;
	}
}


TEST(PERCEPTRON, layer_multiple_in) {
	nnet::session& sess = nnet::session::get_instance();
	nnet::placeholder<double>* in1 = nnet::placeholder<double>::make((std::vector<size_t>{5}), "layerin1");
	nnet::placeholder<double>* in2 = nnet::placeholder<double>::make((std::vector<size_t>{5}), "layerin2");
	nnet::layer_perceptron layer = nnet::layer_perceptron(5, 5);
	nnet::ivariable<double>* res1 = layer(in1);
	nnet::ivariable<double>* res2 = layer(in2);
	*in1 = std::vector<double>{1, 4, 8, 16, 32};
	*in2 = std::vector<double>{5, 4, 3, 2, 1};
    nnet::expose<double>* e1 = new nnet::expose<double>(res1);
    nnet::expose<double>* e2 = new nnet::expose<double>(res2);
	sess.initialize_all<double>();
	std::vector<double> raw1 = e1->get_raw();
	std::vector<double> raw2 = e2->get_raw();
	ASSERT_EQ(raw1.size(), raw2.size());
	for (size_t i = 0; i < raw1.size(); i++) {
		EXPECT_NE(raw1[i], raw2[i]);
	}
}


TEST(PERCEPTRON, layer_action) {
	nnet::session& sess = nnet::session::get_instance();
	std::vector<double> vin = {1, 2, 3, 4, 5};
	std::vector<double> exout = {1, 2, 3};
	nnet::layer_perceptron layer(vin.size(), exout.size());
	nnet::placeholder<double>* var = nnet::placeholder<double>::make(std::vector<size_t>{5}, "layerin");
	nnet::ivariable<double>* res = layer(var);
	nnet::expose<double>* ex = nnet::expose<double>::make(res);
	// initialize variables
	sess.initialize_all<double>();
	*var = vin;
	std::vector<double> raw = ex->get_raw();
	ASSERT_EQ(raw.size(), exout.size());
}


TEST(PERCEPTRON, mlp_multiple_in) {
	nnet::session& sess = nnet::session::get_instance();
	nnet::placeholder<double>* in1 = nnet::placeholder<double>::make((std::vector<size_t>{5}), "layerin1");
	nnet::placeholder<double>* in2 = nnet::placeholder<double>::make((std::vector<size_t>{5}), "layerin2");
	std::vector<IN_PAIR> hiddens = {
			// use same sigmoid in static memory once deep copy is established
			IN_PAIR(5, nnet::sigmoid<double>),
			IN_PAIR(5, nnet::sigmoid<double>),
			IN_PAIR(5, nnet::sigmoid<double>),
	};
	nnet::ml_perceptron mlp = nnet::ml_perceptron(5, hiddens);
	nnet::ivariable<double>* res1 = mlp(in1);
	nnet::ivariable<double>* res2 = mlp(in2);
	*in1 = std::vector<double>{1, 4, 8, 16, 32};
	*in2 = std::vector<double>{5, 4, 3, 2, 1};
	nnet::expose<double>* e1 = nnet::expose<double>::make(res1);
	nnet::expose<double>* e2 = nnet::expose<double>::make(res2);
	sess.initialize_all<double>();
	std::vector<double> raw1 = e1->get_raw();
	std::vector<double> raw2 = e2->get_raw();
	ASSERT_EQ(raw1.size(), raw2.size());
	for (size_t i = 0; i < raw1.size(); i++) {
		EXPECT_NE(raw1[i], raw2[i]);
	}
}


TEST(PERCEPTRON, mlp_action) {
	nnet::session& sess = nnet::session::get_instance();
	size_t n_out = 3;
	size_t n_hidden = 4;
	std::vector<double> vin = {1, 2, 3, 4, 5};
	// we keep ownership of activation functions
	std::vector<IN_PAIR> hiddens = {
		// use same sigmoid in static memory once deep copy is established
		IN_PAIR(n_hidden, nnet::sigmoid<double>),
		IN_PAIR(n_hidden, nnet::sigmoid<double>),
		IN_PAIR(n_out, nnet::sigmoid<double>),
	};
	// mlp has ownership of activations
	nnet::ml_perceptron mlp = nnet::ml_perceptron(vin.size(), hiddens);

	nnet::placeholder<double>* var = nnet::placeholder<double>::make((std::vector<size_t>{5}), "layerin");
	nnet::ivariable<double>* res = mlp(var);
	nnet::expose<double>* ex = nnet::expose<double>::make(res);
	sess.initialize_all<double>();
	*var = vin;
	std::vector<double> raw = ex->get_raw();
	ASSERT_EQ(raw.size(), n_out);
	double sum = 0;
	for (double o : raw) {
		sum += o;
		EXPECT_LE(o, 1);
		EXPECT_GE(o, 0);
	}
	ASSERT_GT(sum/raw.size(), 0);
}


TEST(PERCEPTRON, layer_optimizer) {
	nnet::OPTIMIZER<double> optimizer = std::make_shared<nnet::gd_optimizer>(0.001);
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

	nnet::ml_perceptron mlp = nnet::ml_perceptron(n_in, hiddens);
	nnet::placeholder<double>* var = nnet::placeholder<double>::make((std::vector<size_t>{10}), "layerin");
	nnet::ivariable<double>* res = mlp(var);
	nnet::placeholder<double>* out = nnet::placeholder<double>::make((std::vector<size_t>{5}), "expectout");
	nnet::ivariable<double>* diff = res - std::static_pointer_cast<nnet::ivariable<double> >(out);

	*var = std::vector<double>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1,};
	*out = std::vector<double>{1, 1, 1, 1, 1,};

	// SEE THE GRADIENTS
	nnet::GRAD_MAP<double> grads = optimizer->compute_grad(diff);
	sess.initialize_all<double>();

	for (auto gpair : grads) {
		nnet::ivariable<double>* leaf = gpair.first;
		nnet::ivariable<double>* grad_wrt_leaf = gpair.second;
		nnet::expose<double>* ex_grad = nnet::expose<double>::make(grad_wrt_leaf);
		std::vector<double> out = ex_grad->get_raw();
		for (double gres : out) {
			std::cout << gres << " ";
		} std::cout << std::endl;
	}
}


// test with gradient descent
TEST(PERCEPTRON, gd_train) {
//	nnet::session& sess = nnet::session::get_instance();
//	sess.seed_rand_eng(1);
//	size_t n_in = 10;
//	size_t n_out = n_in/2;
//	size_t n_hidden = 8;
//	std::vector<IN_PAIR> hiddens = {
//		// use same sigmoid in static memory once deep copy is established
//		IN_PAIR(n_hidden, nnet::sigmoid<double>),
//		IN_PAIR(n_hidden, nnet::sigmoid<double>),
//		IN_PAIR(n_out, nnet::sigmoid<double>),
//	};
//	size_t batch_size = 1;
//	size_t test_size = 100;
//	size_t train_size = 10000;
//	nnet::gd_net net(n_in, hiddens);
//	net.set_the_record_str8(true);
//	nnet::placeholder<double>* fanin = nnet::placeholder<double>::make((std::vector<size_t>{n_in, batch_size}), "fanin");
//	nnet::ivariable<double>* fanout = net(fanin);
//	nnet::expose<double>* exposeout = nnet::expose<double>::make(fanout);
//	sess.initialize_all<double>();
//
//	std::vector<VECS> samples;
//	for (size_t i = 0; i < train_size; i++) {
//		std::cout << "training " << i << "\n";
//		samples.clear();
//		fill_binary_samples(samples, n_in, batch_size);
//		net.train(samples[0].first, samples[0].second);
//	}
//
//	size_t fails = 0;
//	double err = 0;
//	for (size_t i = 0; i < test_size; i++) {
//		std::cout << "testing " << i << "\n";
//		samples.clear();
//		fill_binary_samples(samples, n_in, test_size);
//
//		*fanin = samples[0].first;
//		std::vector<double> res = exposeout->get_raw();
//		std::vector<double> expect = samples[0].second;
//		for (size_t i = 0; i < res.size(); i++) {
//			err += std::abs(expect[i] - res[i]);
//			if (expect[i] != std::round(res[i])) {
//				fails++;
//			}
//		}
//	}
//	double successrate = 1.0-(double)fails/(test_size*n_out);
//	err /= (test_size*n_out);
//	ASSERT_GE(successrate, 0.80); // TODO increase to 90
//	std::cout << "average err: " << err << std::endl;
//	std::cout << "success rate: " << successrate << std::endl;
}


// test with gradient descent optimizer
TEST(PERCEPTRON, gd_optimizer_train) {
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
	size_t train_size = 10000;

	nnet::OPTIMIZER<double> optimizer = std::make_shared<nnet::gd_optimizer>(0.001);
	nnet::gd_net net(n_in, hiddens, optimizer);
	nnet::placeholder<double>* fanin = nnet::placeholder<double>::make((std::vector<size_t>{n_in, batch_size}), "fanin");
	nnet::ivariable<double>* fanout = net(fanin);
	nnet::expose<double>* exposeout = nnet::expose<double>::make(fanout);
	sess.initialize_all<double>();
	net.set_the_record_str8(true);

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

		*fanin = samples[0].first;
		std::vector<double> res = exposeout->get_raw();
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


// test with batch gradient descent
TEST(PERCEPTRON, bgd_train) {
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
	size_t batch_size = 12;
	size_t test_size = 100;
	size_t train_size = 5000;
	nnet::gd_net net(n_in, hiddens);
	nnet::placeholder<double>* fanin = nnet::placeholder<double>::make((std::vector<size_t>{n_in}), "fanin");
	nnet::ivariable<double>* fanout = net(fanin);
	nnet::expose<double>* exposeout = nnet::expose<double>::make(fanout);
	sess.initialize_all<double>();

	std::vector<VECS> samples;
	for (size_t i = 0; i < train_size; i++) {
		std::cout << "training " << i << "\n";
		samples.clear();
		fill_binary_samples(samples, n_in, batch_size);
		std::vector<double> first;
		std::vector<double> second;
		flatten(samples, first, second);
		net.train(first, second);
	}

	size_t fails = 0;
	double err = 0;
	for (size_t i = 0; i < test_size; i++) {
		std::cout << "testing " << i << "\n";
		samples.clear();
		fill_binary_samples(samples, n_in, batch_size);
		std::vector<double> first;
		std::vector<double> second;

		flatten(samples, first, second, 0);
		*fanin = first;

		std::vector<double> res = exposeout->get_raw();
		std::vector<double> expect = second;

		for (size_t i = 0; i < res.size(); i++) {
			err += std::abs(expect[i] - res[i]);
			if (expect[i] != std::round(res[i])) {
				fails++;
			}
		}
	}
	double successrate = 1.0-(double)fails/(test_size*n_out);
	err /= (test_size*n_out);
	// at least a weak classifier
	ASSERT_GE(successrate, 0.55); // TODO: increase to 0.9
	std::cout << "average err: " << err << std::endl;
	std::cout << "success rate: " << successrate << std::endl;
}
