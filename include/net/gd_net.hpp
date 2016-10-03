//
//  gd_net.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <unordered_set>
#include "layer.hpp"

#pragma once
#ifndef gd_net_hpp
#define gd_net_hpp

namespace nnet {

// wrapper for
// gradient descent
// TODO transform into a tensor operation "optimizer" similar to tf.optimizer
// look for good optimization algorithms that auto determine good learning rates
// and other parameters to minimize training issues
class gd_net : public ml_perceptron {
	private:
		std::unordered_set<ivariable<double>*> ownership;
		double learning_rate = 0.5; // implement setter
		void clear_ownership (void); // for book keeping, remove when replaced with smart ptrs

	public:
		gd_net (size_t n_input,
			std::vector<IN_PAIR> hiddens,
			std::string scope = "MLP");
		virtual ~gd_net (void);
		// operator () is inherited from ml_perceptron
		void train (ivariable<double>& expected_out);

		// DEPRECATED
		gd_net (
			size_t n_input,
			std::vector<std::pair<size_t,
			adhoc_operation> > hiddens,
			std::string scope = "MLP") : ml_perceptron(n_input, hiddens, scope) {}

		// gradient descent for linear regression
		void train (VECS io_pair) {
			std::vector<VECS> samples = {io_pair};
			train(samples);
		}

		// batch gradient descent
		// prone to overfitting
		void train (std::vector<VECS> sample);
};

}

#endif /* gd_net_hpp */
