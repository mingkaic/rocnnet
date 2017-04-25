//
//  gd_net.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "gd_net.hpp"

#ifdef gd_net_hpp

using namespace nnet;

namespace rocnnet
{

gd_net::gd_net (size_t n_input, std::vector<IN_PAIR> hiddens, std::string scope) :
	ml_perceptron(n_input, hiddens, scope)
{
	size_t n_output = hiddens.back().first;
	train_in_ = new nnet::placeholder<double>(
		std::vector<size_t>{n_input, 0}, "train_in");
	expected_out_ = new nnet::placeholder<double>(
		std::vector<size_t>{n_output, 0}, "expected_out");
	train_setup();
}

// batch gradient descent
// 1/m*sum_m(Err(X, Y)) once matrix operations support auto derivation
// then apply cost funct to grad desc alg:
// new weight = old weight - learning_rate * cost func gradient over old weight
// same thing with bias (should experience no rocnnet decrease due to short circuiting)
void gd_net::train (std::vector<double>& train_in, std::vector<double>& expected_out)
{
	*train_in_ = train_in;
	*expected_out_ = expected_out;
	error_->update_status(true); // freeze
	for (auto& trainer : updates_)
	{
		trainer();
	}
	error_->update_status(false); // update again
}

void gd_net::train_setup (void)
{
	nnet::varptr<double> output = ml_perceptron::operator()(train_in_);
	nnet::varptr<double> diff = nnet::varptr<double>(expected_out_) - output;
	nnet::varptr<double> error = diff * diff;
	error_ = static_cast<nnet::iconnector<double>*>(error.get());

	nnet::inode<double>::GRAD_CACHE leafset;
	error->get_leaves(leafset);
	for (auto lit : leafset)
	{
		nnet::variable<double>* Wb = lit.first;
		nnet::varptr<double> gres = error->get_gradient(Wb);
		// Wb = Wb + learning_rate * gres
		updates_.push_back(Wb->assign_add(gres * learning_rate_));
	}
}

}

#endif /* gd_net_hpp */
