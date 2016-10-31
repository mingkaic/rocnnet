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

#define EXPOSE_PTR std::shared_ptr<nnet::expose<double> >

// wrapper for
// gradient descent
// TODO transform into a tensor operation "optimizer" similar to tf.optimizer
// look for good optimization algorithms that auto determine good learning rates
// and other parameters to minimize training issues
class gd_net : public ml_perceptron {
	private:
		size_t n_input;
		double learning_rate = 0.5; // implement setter
		bool record_training = false;
		// input
		PLACEHOLDER_PTR<double> train_in = nullptr;
		PLACEHOLDER_PTR<double> expected_out = nullptr;
		PLACEHOLDER_PTR<double> batch_size = nullptr;
		// output
		EVOKER_PTR<double> updates;
		EXPOSE_PTR record = nullptr;

		void train_set_up (nnet::OPTIMIZER<double> optimizer);
		gd_net (const gd_net& net, std::string scope);

	public:
		gd_net (size_t n_input,
			std::vector<IN_PAIR> hiddens,
			nnet::OPTIMIZER<double> optimizer = nullptr,
			std::string scope = "MLP");
		virtual ~gd_net (void) {}
		virtual gd_net* clone (std::string scope = "MLP_COPY") { return new gd_net(*this, scope); }

		void set_the_record_str8 (bool record_training) {
			this->record_training = record_training;
		}

		// operator () is inherited from ml_perceptron
		void train (std::vector<double> train_in,
					std::vector<double> expected_out);
};

}

#endif /* gd_net_hpp */
