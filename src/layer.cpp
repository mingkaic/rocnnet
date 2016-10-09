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
const_init<double> layer_perceptron::oinit(1);

void layer_perceptron::copy (
	layer_perceptron const & other,
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
		layer_perceptron const & other,
		std::string scope) {
	copy(other, scope);
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
	std::string scope)
	: n_input(n_input), n_output(n_output) {
	// inputs pipe into the rows of the weight
	weights = new variable<double>(
		std::vector<size_t>{n_output, n_input},
		rinit, scope+"_weights");
	bias = new variable<double>(
		std::vector<size_t>{n_output},
		zinit, scope+"_bias");
}

layer_perceptron::~layer_perceptron (void) {
	if (weights) {
		delete weights;
		delete bias;
	}
	clear_ownership();
}

layer_perceptron& layer_perceptron::operator = (const layer_perceptron& other) {
	if (&other != this) {
		if (weights) {
			delete weights;
			delete bias;
		}
		copy(other, scope);
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
// input are expected to have shape n_input by batch_size
// outputs are expected to have shape output by batch_size
ivariable<double>& layer_perceptron::operator () (
	ivariable<double>& input) {
	// weights are n_output column by n_input rows
	tensor_shape ts = input.get_shape();
	assert(2 >= ts.n_dims() && ts.is_fully_defined());
	// batch size is 1 if tensor is only vector
	size_t batch_size = 1 == ts.n_dims() ? 1 : ts.as_list()[1];
	ivariable<double>* mres = new matmul<double>(input, *weights);
	// mres is n_output column by batch_size rows (extend bias to fit)
	// TODO: replace with extend
	variable<double>* extension =
		new variable<double>(std::vector<size_t>{1, batch_size}, oinit);
	ivariable<double>* exbias = new matmul<double>(*extension, *bias);
	ivariable<double>* res = new add<double>(*mres, *exbias);
	// take ownership of all variables
	clear_ownership(); // clear room for new ownership
	ownership = {mres, extension, exbias, res};
	return *res;
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
}

ml_perceptron& ml_perceptron::operator = (const ml_perceptron& other) {
	if (&other != this) {
		for (HID_PAIR hp : layers) {
			delete hp.first;
			delete hp.second;
		}
		copy(other, scope);
	}
	return *this;
}

// input are expected to have shape n_input by batch_size
// outputs are expected to have shape output by batch_size
ivariable<double>& ml_perceptron::operator () (placeholder<double>& input) {
	ivariable<double>* output;
	if (nullptr == in_place) {
		// (activation functions, hp.second, can only be bound once)
		// output of one layer's dimensions is expected to be matched by
		// the layer_perceptron of the next layer
		in_place = &input;
		output = &input;
		for (HID_PAIR hp : layers) {
			std::vector<size_t> shapes = output->get_shape().as_list();
			ivariable<double> &hypothesis = (*hp.first)(*output);
			hypothesi.push_back(&hypothesis);
			output = &(*hp.second)(hypothesis);
		}
	} else {
		// input takes over all in_place consumers
		in_place->replace(input);
		in_place = &input;
		// last layer owns output
		output = &(*(layers.rbegin())->second);
	}
	return *output;
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
