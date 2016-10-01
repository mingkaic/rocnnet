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

#include "../utils.hpp"
#include "../graph.hpp"

#include <iostream>
#include <cassert>

#pragma once
#ifndef layer_hpp
#define layer_hpp

namespace nnet {

#define V_MATRIX std::vector<std::vector<double> > // replace with tensor
#define VECS std::pair<std::vector<double>, std::vector<double> >

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

		layer_perceptron& operator = (layer_perceptron const & other);

		ivariable<double>* operator () (ivariable<double> & input);

		size_t get_n_input (void) const { return n_input; }
		size_t get_n_output (void) const { return n_output; }
		WB_PAIR get_variables (void) const { return WB_PAIR(weights, bias); }

		// DEPRECATED
		V_MATRIX* raw_weights = nullptr;
		double* raw_bias = nullptr;
		adhoc_operation op;
		layer_perceptron (
			size_t n_input,
			size_t n_output,
			adhoc_operation op,
			std::string scope="");
		std::vector<double> operator () (std::vector<double> const & input);
		std::vector<double> hypothesis (std::vector<double> const & input);
		// expose raw_weights and raw_bias
		// TODO: replace array with tensors
		std::pair<V_MATRIX&, double*> get_vars (void) {
			return std::pair<V_MATRIX&, double*>(*raw_weights, raw_bias);
		}
};

class ml_perceptron {
	private:
		std::string scope;
		std::vector<HID_PAIR> layers;

		void copy (
			ml_perceptron const & other,
			std::string scope);

	public:
		// trust that passed in operations are unconnected
		ml_perceptron (
			size_t n_input,
			std::vector<IN_PAIR> hiddens,
			std::string scope = "MLP");
		ml_perceptron (
			ml_perceptron const & other,
			std::string scope="");
		virtual ~ml_perceptron (void);
		ml_perceptron& operator = (ml_perceptron const & other);

		ivariable<double>* operator () (ivariable<double> & input);
		std::vector<WB_PAIR> get_variables (void);

		// def get_var(self):
		// 	# return the ultimate layer value
		// 	res = self.input_layer.get_var()
		// 	for inner in self.layers:
		// 		res.extend(inner.get_var())
		// 	return res

		// DEPRECATED
		std::vector<layer_perceptron*> raw_layers;
		ml_perceptron (
			size_t n_input,
			std::vector<std::pair<size_t,
			adhoc_operation> > hiddens,
			std::string scope = "MLP");
		std::vector<double> operator () (std::vector<double> const & input);
		std::vector<layer_perceptron*> get_vars (void) {
			return raw_layers;
		}
};

}

#endif /* layer_hpp */
