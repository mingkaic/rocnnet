//
//  layer.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "../include/layer.hpp"

#ifdef layer_hpp

namespace nnet {

// LAYER PERCEPTRON IMPLEMENTATION

random_uniform<double> layer_perceptron::rinit(-1, 1);
const_init<double> layer_perceptron::zinit(0);

void layer_perceptron::copy (
	const layer_perceptron& other,
	std::string scope) {
	n_input = other.n_input;
	n_output = other.n_output;
	if (0 == scope.size()) {
		scope = other.scope + "_cpy";
	}
	this->scope = scope;
	weights_ = other.weights_->clone(scope);
	bias_ = other.bias_->clone(scope);
}

layer_perceptron::layer_perceptron (
		const layer_perceptron& other,
		std::string scope) {
	this->copy(other, scope);
}

// weights are <output, input>
layer_perceptron::layer_perceptron (size_t n_input, size_t n_output, std::string scope) : 
		n_input(n_input), n_output(n_output) {
	weights_ = new variable<double>(std::vector<size_t>{n_output, n_input}, rinit, scope+"_weights");
	bias_ = new variable<double>(std::vector<size_t>{n_output}, zinit, scope+"_bias");
}

layer_perceptron& layer_perceptron::operator = (const layer_perceptron& other) {
	if (&other != this) {
		this->copy(other, scope);
	}
	return *this;
}

// input are expected to have shape n_input by batch_size
// outputs are expected to have shape output by batch_size
ivariable<double>* layer_perceptron::operator () (ivariable<double>* input) {
	// weights are n_output column by n_input rows
	varptr<double> mres = new matmul<double>(input, weights_));
	varptr<double> bias = fit(bias_, mres); // adjust shape based on mres shape
	return mres + bias;
}

// MULTILAYER PERCEPTRON IMPLEMENTATION

void ml_perceptron::copy (
	const ml_perceptron& other,
	std::string scope) {
	if (0 == scope.size()) {
		scope = other.scope + "_cpy";
	}
	this->scope = scope;
	size_t level = 0;
	for (HID_PAIR hp : other.layers) {
		layers.push_back(HID_PAIR(
			new layer_perceptron(*hp.first,
				nnutils::formatter() << scope << ":hiddens" << level++),
				hp.second));
	}
}

ml_perceptron::ml_perceptron (
	size_t n_input,
	std::vector<IN_PAIR> hiddens,
	std::string scope) : scope(scope) {
	size_t level = 0;
	size_t n_output;
	layer_perceptron* layer;
	for (IN_PAIR ip : hiddens) {
		n_output = ip.first;
		layer = new layer_perceptron(n_input, n_output,
			nnutils::formatter() << scope << ":hidden_" << level++);
		layers.push_back(HID_PAIR(layer, ip.second));
		n_input = n_output;
	}
}

ml_perceptron::ml_perceptron (
	const ml_perceptron& other,
	std::string scope) {
	this->copy(other, scope);
}

ml_perceptron::~ml_perceptron (void) {
	for (HID_PAIR hp : layers) {
		delete hp.first;
	}
}

ml_perceptron& ml_perceptron::operator = (const ml_perceptron& other) {
	if (&other != this) {
		for (HID_PAIR hp : layers) {
			delete hp.first;
		}
		this->copy(other, scope);
	}
	return *this;
}

// input are expected to have shape n_input by batch_size
// outputs are expected to have shape output by batch_size
ivariable<double>* ml_perceptron::operator () (placeholder<double>* input) {
	// output of one layer's dimensions is expected to be matched by
	// the layer_perceptron of the next layer
	varptr<double> output = input;
	for (HID_PAIR hp : layers) {
		varptr<double> hypothesis = (*hp.first)(output);
		output = (hp.second)(hypothesis);
	}
	return output;
}

std::vector<WB_PAIR> ml_perceptron::get_variables (void) {
	std::vector<WB_PAIR> wb_vec;
	for (HID_PAIR hp : layers) {
		wb_vec.push_back(hp.first->get_variables());
	}
	return wb_vec;
}

}

#endif
