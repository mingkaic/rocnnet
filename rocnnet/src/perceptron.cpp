//
//  layer.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "perceptron.hpp"

#ifdef ROCNNET_PERCEPTRON_HPP

namespace rocnnet
{

perceptron::perceptron (size_t n_input, size_t n_output, std::string scope) :
	n_input(n_input), n_output(n_output)
{
	nnet::rand_uniform<double> rinit(-1, 1);
	nnet::const_init<double> zinit(0);
	weights_ = new nnet::variable<double>(std::vector<size_t>{n_output, n_input},
		rinit, scope+"_weights");
	bias_ = new nnet::variable<double>(std::vector<size_t>{n_output},
		zinit, scope+"_bias");
}

perceptron::~perceptron (void)
{
	delete weights_;
	delete bias_;
}

perceptron::perceptron (
	const perceptron& other,
	std::string scope)
{
	copy_helper(other, scope);
}

perceptron::perceptron (
	perceptron&& other)
{
	move_helper(std::move(other));
}

perceptron& perceptron::operator = (const perceptron& other)
{
	if (&other != this)
	{
		this->copy_helper(other, scope + "_cpy");
	}
	return *this;
}

perceptron& perceptron::operator = (perceptron&& other)
{
	if (&other != this)
	{
		this->move_helper(std::move(other));
	}
	return *this;
}

// input are expected to have shape n_input by batch_size
// outputs are expected to have shape n_output by batch_size
nnet::varptr<double> perceptron::operator () (nnet::inode<double>* input)
{
	// weights are n_output column by n_input rows
	nnet::varptr<double> weighed = nnet::matmul<double>::get(input, weights_);
	// bias are natively n_output column by 1 rows
	// todo: replace fit operator by mappable
	nnet::varptr<double> bias = nnet::fit<double>(bias_, weighed); // adjust shape based on mres shape
	return weighed + bias;
}

size_t perceptron::get_n_input (void) const { return n_input; }

size_t perceptron::get_n_output (void) const { return n_output; }

WB_PAIR perceptron::get_variables (void) const { return WB_PAIR(weights_, bias_); }

void perceptron::copy_helper (const perceptron& other, std::string scope)
{
	n_input = other.n_input;
	n_output = other.n_output;
	this->scope = scope;
	weights_ = other.weights_->clone();
	bias_ = other.bias_->clone();
	weights_->set_label(scope+"_weights");
	bias_->set_label(scope+"_bias");
}

void perceptron::move_helper (perceptron&& other)
{
	n_input = std::move(other.n_input);
	n_output = std::move(other.n_output);
	scope = std::move(other.scope);
	weights_ = other.weights_->move();
	bias_ = other.bias_->move();
}

}

#endif
