//
//  gd_net.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "../include/gd_net.hpp"

#ifdef gd_net_hpp

namespace nnet {

void gd_net::train_set_up (void) {
	// follow () operator for ml_perceptron except store hypothesis
	std::queue<VAR_PTR<double> > layer_out;
	std::stack<VAR_PTR<double> > prime_out;

	VAR_PTR<double> output = std::dynamic_pointer_cast<ivariable<double> >(train_in);
	for (HID_PAIR hp : layers) {
		VAR_PTR<double> hypothesis = (*hp.first)(output);
		output = (hp.second)(hypothesis);
		layer_out.push(output);
		// act'(z_i)
//		VAR_PTR<double> s = sigmoid(hypothesis);
//		VAR_PTR<double> grad = s*(1.0-s);
		VAR_PTR<double> grad = gradient<double>::make(output, hypothesis);
		prime_out.push(grad);
	}

	// err_n = (out-y)*act'(z_n)
	// where z_n is the input to the nth layer (last)
	// and act is the activation operation
	VAR_PTR<double> diff = output - PLACEHOLDER_TO_VAR<double>(expected_out);
	record = expose<double>::make(diff);
	VAR_PTR<double> err = diff * prime_out.top();
	prime_out.pop();
	std::stack<VAR_PTR<double> > errs;
	errs.push(err);

	auto term = --this->layers.rend();
	for (auto rit = this->layers.rbegin();
		 term != rit; rit++) {
		HID_PAIR hp = *rit;
		// err_i = matmul(err_i+1, transpose(weight_i))*f'(z_i)
		VAR_PTR<double> weight_i = hp.first->get_variables().first;
		// weight is input by output, err is output by batch size, so we expect mres to be input by batch size
		VAR_PTR<double> mres = matmul<double>::make(err, weight_i, false ,true);
		err = mres * prime_out.top();
		prime_out.pop();
		errs.push(err);
	}

	VAR_PTR<double> learn_batch = learning_rate / PLACEHOLDER_TO_VAR<double>(batch_size);
	output = train_in;
	for (HID_PAIR hp : this->layers) {
		err = errs.top();
		errs.pop();

		// dweights = learning*matmul(transpose(layer_in), err)
		// dbias = learning*err
		VAR_PTR<double> cost = matmul<double>::make(output, err, true);
		VAR_PTR<double> dweights = cost * learn_batch;

		// expecting err to be output by batchsize, compress along batchsize
		VAR_PTR<double> compressed_err = compress<double>::make(err, 1);
		VAR_PTR<double> dbias = compressed_err * learn_batch;

		// update weights and bias
		// weights -= dweights, bias -= dbias
		WB_PAIR wb = hp.first->get_variables();
		EVOKER_PTR<double> w_evok = std::make_shared<update_sub<double> >(
			std::static_pointer_cast<variable<double>, ivariable<double> >(wb.first), dweights);
		EVOKER_PTR<double> b_evok = std::make_shared<update_sub<double> >(
			std::static_pointer_cast<variable<double>, ivariable<double> >(wb.second), dbias);

		// store for update
		updates.push_back(w_evok);
		updates.push_back(b_evok);

		output = layer_out.front();
		layer_out.pop();
	}
}

gd_net::gd_net (const gd_net& net, std::string scope) : ml_perceptron(net, scope) {
	n_input = net.n_input;
	learning_rate = net.learning_rate;
	batch_size = net.batch_size->clone();
	train_in = net.train_in->clone();
	expected_out = net.expected_out->clone();
	train_set_up();
}

gd_net::gd_net (size_t n_input,
	std::vector<IN_PAIR> hiddens,
	std::string scope)
	: ml_perceptron(n_input, hiddens, scope), n_input(n_input) {
	size_t n_out = hiddens.back().first;
	batch_size = placeholder<double>::make(std::vector<size_t>{0}, "batch_size");
	train_in = placeholder<double>::make(std::vector<size_t>{n_input, 0}, "train_in");
	expected_out = placeholder<double>::make(std::vector<size_t>{n_out, 0}, "expected_out");
	train_set_up();
}

// batch gradient descent
// TODO upgrade to using ioperation's gradient method for cost function (determine speedup before "upgrade")
// 1/m*sum_m(Err(X, Y)) once matrix operations support auto derivation
// then apply cost funct to grad desc alg:
// new weight = old weight - learning_rate * cost func gradient over old weight
// same thing with bias (should experience no rocnnet decrease due to short circuiting)
void gd_net::train (std::vector<double> train_in,
					std::vector<double> expected_out) {
	(*this->batch_size) = std::vector<double>{(double) train_in.size() / n_input};
	(*this->train_in) = train_in;
	(*this->expected_out) = expected_out;
	// trigger update
	for (EVOKER_PTR<double> evoks : updates) {
		evoks->eval();
	}
	if (record_training) {
		std::vector<double> errs = record->get_raw();
		double avg_err = 0;
		for (double e : errs) {
			avg_err += std::abs(e);
		}
		avg_err /= errs.size();
		std::cout << "err: " << avg_err << "\n";
	}
}

}

#endif /* gd_net_hpp */
