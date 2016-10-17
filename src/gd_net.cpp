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

void gd_net::train (tensor_shape ts) {
	clear_ownership(); // all intermediate vars are used once anyways

	// follow () operator for ml_perceptron except store hypothesis
	std::stack<ivariable<double>*> act_prime;
	ivariable<double>* output = this->in_place;
	for (size_t i = 0; i < hypothesi.size(); i++) {
		// act'(z_i)
		ivariable<double>* prime = new gradient<double>(*this->layers[i].second, *hypothesi[i]);
		ownership.emplace(prime);
		act_prime.push(prime);
		output = this->layers[i].second;
	}
	output->eval(); // run data through weights once before evaluating updated weights
	// err_n = (out-y)*act'(z_n)
	// where z_n is the input to the nth layer (last)
	// and act is the activation operation
	// take ownership
	expected_out = new placeholder<double>(ts);
	ivariable<double>* diff = new sub<double>(*output, *expected_out);
	ivariable<double>* err = new mul<double>(*diff, *act_prime.top());

	ownership.emplace(expected_out);
	ownership.emplace(diff);
	ownership.emplace(err);
	act_prime.pop();

	std::stack<ivariable<double>*> errs;
	errs.push(err);
	auto term = --this->layers.rend();
	for (auto rit = this->layers.rbegin();
		 term != rit; rit++) {
		HID_PAIR hp = *rit;
		// err_i = matmul(err_i+1, transpose(weight_i))*f'(z_i)
		ivariable<double>* weight_i = hp.first->get_variables().first;
		// weight is input by output, err is output by batch size, so we expect mres to be input by batch size
		ivariable<double>* mres = new matmul<double>(*err, *weight_i, false ,true);
		err = new mul<double>(*mres, *act_prime.top());
		act_prime.pop();
		errs.push(err);
		ownership.emplace(mres);
		ownership.emplace(err);
	}

	double learn_batch = learning_rate / ts.as_list()[1];
	output = this->in_place;
	for (HID_PAIR hp : this->layers) {
		err = errs.top();
		errs.pop();

		// dweights = learning*matmul(transpose(layer_in), err)
		// dbias = learning*err
		ivariable<double>* cost = new matmul<double>(*output, *err, true);
		ivariable<double>* dweights = new mul<double>(*cost, learn_batch);

		// expecting err to be output by batchsize, compress along batchsize
		ivariable<double>* compressed_err = new compress<double>(*err, 1);
		ivariable<double>* dbias = new mul<double>(*compressed_err, learn_batch);

		ownership.emplace(cost);
		ownership.emplace(dweights);
		ownership.emplace(compressed_err);
		ownership.emplace(dbias);

		differentials.push_back(IVARS(dweights, dbias));
		// store for update

		output = hp.second;
	}
}

void gd_net::clear_ownership (void) {
	for (ivariable<double>* mine : ownership) {
		delete mine;
	}
	ownership.clear();
	differentials.clear();
	expected_out = nullptr;
}

gd_net::gd_net (const gd_net& net, std::string scope) :
	ml_perceptron(net, scope) {
	learning_rate = net.learning_rate;
	expected_out = net.expected_out;
	differentials = net.differentials;
}

gd_net::gd_net (size_t n_input,
	std::vector<IN_PAIR> hiddens,
	std::string scope)
	: ml_perceptron(n_input, hiddens, scope) {}

gd_net::~gd_net (void) {
	clear_ownership();
}

// batch gradient descent
// TODO upgrade to using ioperation's gradient method for cost function (determine speedup before "upgrade")
// 1/m*sum_m(Err(X, Y)) once matrix operations support auto derivation
// then apply cost funct to grad desc alg:
// new weight = old weight - learning_rate * cost func gradient over old weight
// same thing with bias (should experience no rocnnet decrease due to short circuiting)
void gd_net::train (ivariable<double>& e_out) {
	if (nullptr == expected_out ||
		false == e_out.get_shape().is_compatible_with(
			expected_out->get_shape())) {
		train(e_out.get_shape());
	}
	expected_out->assign(e_out);

	for (size_t i = 0; i < differentials.size(); i++) {
		IVARS ds = differentials[i];
		WB_PAIR wb = layers[i].first->get_variables();
		// update weights and bias
		// weights -= dweights, bias -= dbias
		const tensor<double>& dwt = ds.first->eval();
		const tensor<double>& dbt = ds.second->eval();
		wb.first->update(dwt, shared_cnnet::op_sub<double>);
		wb.second->update(dbt, shared_cnnet::op_sub<double>);
	}
}

}

#endif /* gd_net_hpp */
