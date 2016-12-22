//
//  layer.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "../include/layer.hpp"

#ifdef layer_hpp

namespace nnet
{

// LAYER PERCEPTRON IMPLEMENTATION

random_uniform<double> layer_perceptron::rinit(-1, 1);
const_init<double> layer_perceptron::zinit(0);

void layer_perceptron::copy (const layer_perceptron& other, std::string scope)
{
	n_input = other.n_input;
	n_output = other.n_output;
	this->scope = scope;
	weights_ = other.weights_->clone();
	bias_ = other.bias_->clone();
}

layer_perceptron::layer_perceptron (size_t n_input, size_t n_output, std::string scope) : 
	n_input(n_input), n_output(n_output)
{
	weights_ = new variable<double>(std::vector<size_t>{n_output, n_input}, 
		rinit, scope+"_weights");
	bias_ = new variable<double>(std::vector<size_t>{n_output}, 
		zinit, scope+"_bias");
}

layer_perceptron::~layer_perceptron (void)
{
	delete weights_;
	delete bias_;
}

layer_perceptron::layer_perceptron (
	const layer_perceptron& other,
	std::string scope)
{
	copy(other, scope);
}

layer_perceptron& layer_perceptron::operator = (const layer_perceptron& other)
{
	if (&other != this) {
		this->copy(other, scope);
	}
	return *this;
}

// input are expected to have shape n_input by batch_size
// outputs are expected to have shape output by batch_size
varptr<double> layer_perceptron::operator () (ivariable<double>* input)
{
	// weights are n_output column by n_input rows
	varptr<double> weighed = matmul<double>::build(input, weights_);
	varptr<double> bias = fit<double>(bias_, weighed); // adjust shape based on mres shape
	return weighed + bias;
}

}

#endif
