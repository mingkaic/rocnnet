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
	weights = other.weights->clone(scope);
	bias = other.bias->clone(scope);
}

layer_perceptron::layer_perceptron (
		const layer_perceptron& other,
		std::string scope) {
	copy(other, scope);
}

layer_perceptron::layer_perceptron (
	size_t n_input,
	size_t n_output,
	std::string scope)
	: n_input(n_input), n_output(n_output) {
	// inputs pipe into the rows of the weight
	weights = variable<double>::make(
		std::vector<size_t>{n_output, n_input},
		rinit, scope+"_weights");
	bias = variable<double>::make(
		std::vector<size_t>{n_output},
		zinit, scope+"_bias");
}

layer_perceptron& layer_perceptron::operator = (const layer_perceptron& other) {
	if (&other != this) {
		copy(other, scope);
	}
	return *this;
}

// input are expected to have shape n_input by batch_size
// outputs are expected to have shape output by batch_size
VAR_PTR<double> layer_perceptron::operator () (VAR_PTR<double> input) {
	// weights are n_output column by n_input rows
	VAR_PTR<double> mres = matmul<double>::make(input, weights);
	VAR_PTR<double> exbias = extend<double>::make(bias, mres); // adjust shape based on mres shape
	return mres + exbias;
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
		std::string layername =
			nnutils::formatter() << scope << ":hiddens" << level++;
		layers.push_back(HID_PAIR(
			new layer_perceptron(*hp.first, layername),
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
	ml_perceptron const & other,
	std::string scope) {
	copy(other, scope);
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
		copy(other, scope);
	}
	return *this;
}

// input are expected to have shape n_input by batch_size
// outputs are expected to have shape output by batch_size
VAR_PTR<double> ml_perceptron::operator () (PLACEHOLDER_PTR<double> input) {
	// output of one layer's dimensions is expected to be matched by
	// the layer_perceptron of the next layer
	VAR_PTR<double> output = input;
	for (HID_PAIR hp : layers) {
		VAR_PTR<double> hypothesis = (*hp.first)(output);
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
