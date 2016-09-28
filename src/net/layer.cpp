//
//  layer.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "../../include/nnet.hpp"
#include <iostream>

#ifdef layer_hpp

namespace nnet {

// LAYER PERCEPTRON IMPLEMENTATION

random_uniform<double> layer_perceptron::rinit(-1, 1);
const_init<double> layer_perceptron::zinit(0);
const_init<double> layer_perceptron::oinit(1);

void layer_perceptron::copy (
	layer_perceptron const & other,
	std::string scope) {
	n_input = other.n_input;
	n_output = other.n_output;
	op = other.op;
	if (0 == scope.size()) {
		scope = other.scope + "_cpy";
	}
	this->scope = scope;
	weights = other.weights->clone(scope);
	bias = other.bias->clone(scope);
}

void layer_perceptron::clear_ownership (void) {
	for (ivariable<double>* mine : ownership) {
		delete mine;
	}
	ownership.clear();
}

// def __init__ (self, n_input_arr, n_output, scope="Layer"):
// 	self.scope = scope

// 	# store initializing parameters for copying
// 	self.n_input_arr = n_input_arr
// 	self.n_output = n_output

// 	if type(n_input_arr) != list:
// 		n_input_arr = [n_input_arr]adhoc_operation

// 	# share variable scope
// 	with tf.variable_scope(scope):
// 		self.weights = []
// 		for idx, n_input in enumerate(n_input_arr):
// 			wname = "weight%d" % (idx,)
// 			# rand engine
// 			winit = tf.random_uniform_initializer(-1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input))
// 			self.weights.append(tf.get_var(wname, (n_input, n_output), initializer=winit))
// 		# create a bias
// 		self.bias = tf.get_var("bias", (n_output,), initializer=tf.constant_initializer(0))

layer_perceptron::layer_perceptron (
	size_t n_input,
	size_t n_output,
	std::string scope) {
	// inputs pipe into the rows of the weight
	weights = new variable<double>(
		std::vector<size_t>{n_output, n_input},
		rinit, scope+"_weights");
	bias = new variable<double>(
		std::vector<size_t>{n_output},
		zinit, scope+"_bias");
}

layer_perceptron::layer_perceptron (
	layer_perceptron const & other,
	std::string scope) {
	copy(other, scope);

	// DEPRECATED
	raw_weights = new V_MATRIX();
	raw_bias = new double[n_output];
	for (size_t x = 0; x < n_input; x++) {
		raw_weights->push_back((*other.raw_weights)[x]);
	}
	memcpy(raw_bias, other.raw_bias, n_output*sizeof(double));
}

layer_perceptron::~layer_perceptron (void) {
	if (weights) {
		delete weights;
		delete bias;
	}
	clear_ownership();

	// DEPRECATED
	if (raw_weights) {
		delete raw_weights;
		delete[] raw_bias;
	}
}

layer_perceptron& layer_perceptron::operator = (layer_perceptron const & other) {
	if (&other != this) {
		if (weights) {
			delete weights;
			delete bias;
		}
		copy(other, scope);

		// DEPRECATED
		this->op = other.op;
		delete raw_weights;
		delete[] raw_bias;
		raw_weights = new V_MATRIX();
		raw_bias = new double[n_output];
		for (size_t x = 0; x < n_input; x++) {
			raw_weights->push_back((*other.raw_weights)[x]);
		}
		memcpy(raw_bias, other.raw_bias, n_output*sizeof(double));
	}
	return *this;
}

// def __call__ (self, in_x):
// 	if type(in_x) != list:
// 		in_x = [in_x]
// 	with tf.variable_scope(self.scope):
// 		return sum([tf.matmul(x, weight) for x, weight in zip(in_x, self.weights)]) + self.bias

// returned variable ownership is retained by layer_perceptron instance
// destroys passed out variable from previous calls
// (use with care until smartpointers...)
ivariable<double>* layer_perceptron::operator () (
	ivariable<double>& input) {
	std::vector<size_t> ts = input.get_shape().as_list();
	assert(ts.size() <= 2);
	// input are expected to be batch_size by n_input or n_input by batch_size
	// weights are n_output column by n_input rows,
	// so if ts[0] == n_output then transpose input
	bool transposeA = ts[0] == n_input;
	size_t batch_size = 1;
	if (2 == ts.size()) {
		batch_size = transposeA ? ts[1] : ts[0];
	}
	ivariable<double>* mres = new matmul<double>(input, *weights, transposeA);
	// mres is n_output column by batch_size rows (extend bias to fit)
	variable<double>* extension =
		new variable<double>(std::vector<size_t>{1, batch_size}, oinit);
	ivariable<double>* exbias = new matmul<double>(*extension, *bias);
	ivariable<double>* res = new add<double>(*mres, *exbias);
	// take ownership of all variables
	clear_ownership(); // clear room for new ownership
	ownership = {mres, extension, exbias, res};
	return res;
}

// DEPRECATED
layer_perceptron::layer_perceptron (
	size_t n_input,
	size_t n_output,
	adhoc_operation op,
	std::string scope)
	: n_input(n_input), n_output(n_output), scope(scope) {
	this->op = op;
	raw_weights = new V_MATRIX();
	raw_bias = new double[n_output];

	std::random_device generator;
	std::uniform_real_distribution<double> distribution(0.0,1.0);

	for (size_t x = 0; x < n_input; x++) {
		raw_weights->push_back(std::vector<double>());
		for (size_t y = 0; y < n_output; y++) {
			(*raw_weights)[x].push_back(distribution(generator));
		}
	}
	memset(raw_bias, 0, n_output*sizeof(double));
}

std::vector<double> layer_perceptron::operator () (
	std::vector<double> const & input) {
	std::vector<double> output;
	for (size_t out = 0; out < n_output; out++) {
		output.push_back(0);
		for (size_t in = 0; in < n_input; in++) {
			output[out] += input[in]*(*raw_weights)[in][out];
		}
		output[out] = op(output[out]+raw_bias[out]);
	}
	return output;
}

std::vector<double> layer_perceptron::hypothesis (
	std::vector<double> const & input) {
	std::vector<double> output;
	for (size_t out = 0; out < n_output; out++) {
		output.push_back(0);
		for (size_t in = 0; in < n_input; in++) {
			output[out] += input[in]*(*raw_weights)[in][out];
		}
		output[out] += raw_bias[out];
	}
	return output;
}

// MULTILAYER PERCEPTRON IMPLEMENTATION

void ml_perceptron::copy (
	ml_perceptron const & other,
	std::string scope) {
	if (0 == scope.size()) {
		scope = other.scope + "_cpy";
	}
	this->scope = scope;
	size_t level = 0;
	for (HID_PAIR hp : other.layers) {
		std::string layername =
			nnutils::formatter() << scope << "/hiddens_" << level++;
		layers.push_back(HID_PAIR(
			new layer_perceptron(*hp.first, layername),
			hp.second->clone()));
	}
}

// def __init__ (self, n_input_arr, hiddens, activations, scope="MLP", copy_layers=None):
// 	self.scope = scope
// 	# layer data for copy
// 	self.n_input_arr = n_input_arr
// 	self.hiddens = hiddens
// 	self.input_act, self.inner_act = activations[0], activations[1:]

// 	# share variable scope
// 	with tf.variable_scope(scope):
// 		# copy construct
// 		if copy_layers is not None:
// 			# surface
// 			self.input_layer = copy_layers[0]
// 			# inner layer
// 			self.layers = copy_layers[1:]
// 		else:
// 			self.input_layer = __layer__(n_input_arr, hiddens[0], scope="input_layer")
// 			self.layers = []

// 			# create layers on hidden layers
// 			for l_idx, (l_from, l_to) in enumerate(zip(hiddens[:-1], hiddens[1:])):
// 				self.layers.append(__layer__(l_from, l_to, scope="hidden_layer_%d" % (l_idx,)))
// 	if type(n_input_arr) != list:
// 		self.n_input_arr = [n_input_arr]
// 	if type(hiddens) != list:
// 		self.n_input_arr = [hiddens]
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
			nnutils::formatter() << scope << "/hidden_" << level++);
		layers.push_back(HID_PAIR(layer, ip.second));
		n_input = n_output;
	}
}

// def copy(self, scope=None):
// 	if (None == scope):
// 		scope = self.scope + "_copy"
// 	activations = [self.input_act] + self.inner_act
// 	copy_layer = [self.input_layer.copy()] + [inner.copy() for inner in self.layers]
// 	return MLayerPerceptron(self.n_input_arr, self.hiddens, activations, scope=scope, copy_layers=copy_layer)

ml_perceptron::ml_perceptron (
	ml_perceptron const & other,
	std::string scope) {
	copy(other, scope);
}

ml_perceptron::~ml_perceptron (void) {
	for (HID_PAIR hp : layers) {
		delete hp.first;
		delete hp.second;
	}

	// DEPRECATED
	for (layer_perceptron* pl : raw_layers) {
		delete pl;
	}
}

ml_perceptron& ml_perceptron::operator = (ml_perceptron const & other) {
	if (&other != this) {
		for (HID_PAIR hp : layers) {
			delete hp.first;
			delete hp.second;
		}
		copy(other, scope);

		// DEPRECATED
		for (layer_perceptron* pl : raw_layers) {
			delete pl;
		}
	}
	return *this;
}

ivariable<double>* ml_perceptron::operator () (ivariable<double> & input) {
	// input are expected to be batch_size by n_input or n_input by batch_size
	// output of one layer's dimensions is expected to be matched by
	// the layer_perceptron of the next layer
	ivariable<double>* output = &input;
	for (HID_PAIR hp : layers) {
		ivariable<double>* hypothesis = (*hp.first)(*output);
		output = &(*hp.second)(hypothesis);
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

// def __call__(self, in_x):
// 	if type(in_x) != list:
// 		in_x = [in_x]
// 	with tf.variable_scope(self.scope):
// 		# run input
// 		hidden = self.input_act(self.input_layer(in_x))
// 		# run consecutive layers
// 		for inner, activation in zip(self.layers, self.inner_act):
// 			# activate layer call for x_s = last layer's result
// 			hidden = activation(inner(hidden))
// 		return hidden

// DEPRECATED
ml_perceptron::ml_perceptron (
	size_t n_input,
	std::vector<std::pair<size_t, adhoc_operation> > hiddens,
	std::string scope) : scope(scope) {
	size_t n_output;
	size_t layerlvl = 0;
	for (auto pl : hiddens) {
		n_output = pl.first;
		raw_layers.push_back(
			new layer_perceptron(n_input, n_output, pl.second,
				nnutils::formatter() << scope << "/hidden_" << layerlvl++));
		n_input = n_output;
	}
}

std::vector<double> ml_perceptron::operator () (std::vector<double> const & input) {
	std::vector<double> output = input;
	for (layer_perceptron* lp : raw_layers) {
		output = (*lp)(output);
	}
	return output;
}

}

#endif
