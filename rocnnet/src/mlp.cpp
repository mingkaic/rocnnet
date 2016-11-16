//
//  mlp.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "../include/mlp.hpp"

#ifdef mlp_hpp

namespace nnet
{

// MULTILAYER PERCEPTRON IMPLEMENTATION

void ml_perceptron::copy (const ml_perceptron& other, std::string scope)
{
	for (HID_PAIR hp : layers)
	{
		delete hp.first;
	}
	if (0 == scope.size())
	{
		scope = other.scope + "_cpy";
	}
	this->scope = scope;
	size_t level = 0;
    layer_perceptron* percept;
	for (HID_PAIR hp : other.layers)
	{
	    percept = new layer_perceptron(*hp.first,
			nnutils::formatter() << scope << ":hiddens" << level++);
		layers.push_back(HID_PAIR(percept, hp.second));
	}
}

ml_perceptron::ml_perceptron (const ml_perceptron& other, std::string scope)
{
	this->copy(other, scope);
}

ml_perceptron::ml_perceptron (size_t n_input, std::vector<IN_PAIR> hiddens, 
    std::string scope) : 
    scope(scope)
{
	size_t level = 0;
	size_t n_output;
	layer_perceptron* percept;
	for (IN_PAIR ip : hiddens)
	{
		n_output = ip.first;
		percept = new layer_perceptron(n_input, n_output,
			nnutils::formatter() << scope << ":hidden_" << level++);
		layers.push_back(HID_PAIR(percept, ip.second));
		n_input = n_output;
	}
}

ml_perceptron::~ml_perceptron (void)
{
	for (HID_PAIR hp : layers)
	{
		delete hp.first;
	}
}

ml_perceptron* ml_perceptron::clone (std::string scope)
{
    return new ml_perceptron(*this, scope);
}

ml_perceptron& ml_perceptron::operator = (const ml_perceptron& other)
{
	if (&other != this)
	{
		this->copy(other, scope);
	}
	return *this;
}

// input are expected to have shape n_input by batch_size
// outputs are expected to have shape output by batch_size
varptr<double> ml_perceptron::operator () (placeholder<double>* input)
{
	// output of one layer's dimensions is expected to be matched by
	// the layer_perceptron of the next layer
	ivariable<double>* output = input;
	for (HID_PAIR hp : layers)
	{
		ivariable<double>* hypothesis = (*hp.first)(output);
		output = (hp.second)(hypothesis);
	}
	return output;
}

std::vector<WB_PAIR> ml_perceptron::get_variables (void)
{
	std::vector<WB_PAIR> wb_vec;
	for (HID_PAIR hp : layers)
	{
		wb_vec.push_back(hp.first->get_variables());
	}
	return wb_vec;
}

}

#endif