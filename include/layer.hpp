//
//  layer.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef layer_hpp
#define layer_hpp

#include <string>
#include <vector>
#include <random>
#include <functional>
#include <list>
#include <stack>

#include "graph/bridge/varptr.hpp"
#include "graph/variable/variable.hpp"
#include "graph/operation/special/matmul.hpp"
#include "graph/operation/general/elementary.hpp"

namespace nnet
{

#define WB_PAIR std::pair<variable<double>*, variable<double>*>

// CONSTRAINTS: without tensors, all features are fed by vectors
// higher dimensional features must be contracted to vector or reduced in some manner
// TODO: make convolution neural net via multiple weights per layer
class layer_perceptron
{
	private:
		static random_uniform<double> rinit;
		static const_init<double> zinit;
		static const_init<double> oinit;

		// metadata
		std::string scope;
		size_t n_input;
		size_t n_output;
		
		// content
		// weights have shape <output, input>
		// bias has shape <output>
		variable<double>* weights_ = nullptr;
		variable<double>* bias_ = nullptr;

	protected:
		void copy (const layer_perceptron& other, std::string scope);

	public:
		layer_perceptron (size_t n_input, size_t n_output, std::string scope="");
		// rule of three
		virtual ~layer_perceptron (void);
		layer_perceptron (const layer_perceptron& other, std::string scope="");
		layer_perceptron& operator = (const layer_perceptron& other);

		// input are expected to have shape n_input by batch_size
		// outputs are expected to have shape output by batch_size
		nnet::varptr<double> operator () (varptr<double>);

		size_t get_n_input (void) const { return n_input; }
		size_t get_n_output (void) const { return n_output; }
		WB_PAIR get_variables (void) const { return WB_PAIR(weights_, bias_); }
};

}

#endif /* layer_hpp */
