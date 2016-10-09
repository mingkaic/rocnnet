//
//  layer.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <string>
#include <vector>
#include <random>
#include <functional>
#include <list>
#include <stack>

#include "tenncor/tenncor.hpp"

#include <iostream>
#include <cassert>

#pragma once
#ifndef layer_hpp
#define layer_hpp

namespace nnet {

#define WB_PAIR std::pair<variable<double>*, variable<double>*>
#define IN_PAIR std::pair<size_t, nnet::univar_func<double>*>
#define HID_PAIR std::pair<layer_perceptron*, univar_func<double>*>

// CONSTRAINTS: without tensors, all features are fed by vectors
// higher dimensional features must be contracted to vector or reduced in some manner

class layer_perceptron {
	private:
		static random_uniform<double> rinit;
		static const_init<double> zinit;
		static const_init<double> oinit;

		std::string scope;
		size_t n_input;
		size_t n_output;
		// any allowed size
		variable<double>* weights = nullptr;
		variable<double>* bias = nullptr;
		std::vector<ivariable<double>*> ownership;

		void copy (
			layer_perceptron const & other,
			std::string scope);

		void clear_ownership (void);

	public:
		layer_perceptron (
			size_t n_input,
			size_t n_output,
			std::string scope="");

		layer_perceptron (
			layer_perceptron const & other,
			std::string scope="");

		virtual ~layer_perceptron (void);

		layer_perceptron& operator = (const layer_perceptron& other);

		ivariable<double>& operator () (ivariable<double>& input);

		size_t get_n_input (void) const { return n_input; }
		size_t get_n_output (void) const { return n_output; }
		WB_PAIR get_variables (void) const { return WB_PAIR(weights, bias); }
};

class ml_perceptron {
	//private:
	protected:
		std::string scope;
		std::vector<HID_PAIR> layers;
		placeholder<double>* in_place = nullptr;
		std::vector<ivariable<double>*> hypothesi;

		void copy (ml_perceptron const & other,
			std::string scope);

		ml_perceptron (const ml_perceptron& other, std::string scope);

	public:
		// trust that passed in operations are unconnected
		ml_perceptron (size_t n_input,
			std::vector<IN_PAIR> hiddens,
			std::string scope = "MLP");
		virtual ~ml_perceptron (void);
		ml_perceptron* clone (std::string scope = "MLP_COPY") { return new ml_perceptron(*this, scope); }

		ml_perceptron& operator = (const ml_perceptron& other);

		ivariable<double>& operator () (placeholder<double>& input);
		std::vector<WB_PAIR> get_variables (void);
};

}

#endif /* layer_hpp */
