//
//  gd_net.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <unordered_set>
#include "mlp.hpp"
#include "executor/group.hpp"

#pragma once
#ifndef gd_net_hpp
#define gd_net_hpp

namespace nnet
{

// wrapper for
// gradient descent
// look for good optimization algorithms that auto determine good learning rates
// and other parameters to minimize training issues
class gd_net : public ml_perceptron
{
	private:
		size_t n_input;
		double learning_rate = 0.5; // implement setter
		bool record_training = false;
		// training input
		placeholder<double>* train_in_ = nullptr;
		placeholder<double>* expected_out = nullptr;
		placeholder<double>* batch_size = nullptr;
		// training output
		variable<double>* diff_;
		// training executors
		group<double>* updates;
		// owns optimizer
		ioptimizer<double>* optimizer_ = nullptr;
	
	protected:
		void train_set_up (void);
		
		void copy (const gd_net& other, std::string scope)
		{
			n_input = net.n_input;
			learning_rate = net.learning_rate;
			batch_size = net.batch_size->clone();
			train_in_ = net.train_in_->clone();
			expected_out = net.expected_out->clone();
			train_set_up();
			ml_perceptron<T>::copy(other, scope);
		}
		gd_net (const gd_net& net, std::string scope);

		virtual ml_perceptron* clone_impl (std::string scope)
		{
			return new gd_net(*this, scope);
		}

	public:
		gd_net (size_t n_input,
			std::vector<IN_PAIR> hiddens,
			ioptimizer<double>* optimizer = nullptr,
			std::string scope = "MLP");
		virtual ~gd_net (void) {}
		
		// COPY
		gd_net* clone (std::string scope = "GD_COPY") { return static_cast<gd_net*>(clone_impl(scope)); }
		gd_net& operator = (const gd_net& other)
		{
			if (&other != this)
			{
				copy(other);
			}
			return *this;
		}
		
		// MOVE

		// RECORD TRAINING?
		void set_the_record_str8 (bool record_training)
		{
			this->record_training = record_training;
		}

		// operator () is inherited from ml_perceptron
		void train (std::vector<double> train_in,
					std::vector<double> expected_out);
};

}

#endif /* gd_net_hpp */
