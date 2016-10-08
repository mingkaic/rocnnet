//
//  dqn.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <vector>
#include <cassert>

#include "gd_net.hpp"

#pragma once
#ifndef dqn_hpp
#define dqn_hpp

namespace nnet {

class dq_net {
	private:
		ml_perceptron* q_net;
		ml_perceptron* target_net;
		size_t train_interval;
		double discount_rate;
		double update_rate;

	public:
		dq_net (size_t n_input,
				std::vector<IN_PAIR> hiddens,
				size_t train_interval = 5,
				double discount_rate = 0.95,
				double update_rate = 0.01);

		std::vector<double> operator () (std::vector<double>& input);

		void train (std::vector<std::vector<double> > train_batch);
};

}

#endif /* dqn_hpp */
