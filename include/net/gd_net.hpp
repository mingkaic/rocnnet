//
//  gd_net.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "layer.hpp"

#pragma once
#ifndef gd_net_hpp
#define gd_net_hpp

namespace nnet {

// wrapper for
// gradient descent
struct gd_net {
	ml_perceptron* mlp = NULL;
	double learning_rate = 0.5;

	~gd_net (void) { if (mlp) delete mlp; }

	std::vector<double> operator () (const std::vector<double>& input) {
		if (NULL == mlp) return std::vector<double>();
		return (*mlp)(input);
	}

	// gradient descent for linear regression
	void train (VECS io_pair) {
		if (NULL == mlp) return;
		std::vector<VECS> samples = {io_pair};
		train(samples);
	}

	// batch gradient descent
	// prone to overfitting
	void train (std::vector<VECS> sample);
};

}

#endif /* gd_net_hpp */
