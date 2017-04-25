//
//  gd_net.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <iostream> // for recording
#include <unordered_set>
#include "mlp.hpp"
#include "executor/group.hpp"
#include "executor/optimizer.hpp"

#pragma once
#ifdef ACTIVATE_GD_NET
#ifndef gd_net_hpp
#define gd_net_hpp

namespace rocnnet
{

// wrapper for
// gradient descent
// look for good optimization algorithms that auto determine good learning rates
// and other parameters to minimize training issues
class gd_net : public ml_perceptron
{
public:
	gd_net (size_t n_input,
		std::vector<IN_PAIR> hiddens,
		nnet::ioptimizer<double>* optimizer = nullptr,
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

protected:
	void train_set_up (void);

	void copy (const gd_net& other, std::string scope = "")
	{
		n_input = other.n_input;
		learning_rate = other.learning_rate;
		batch_size = other.batch_size->clone();
		train_in_ = other.train_in_->clone();
		expected_out_ = other.expected_out_->clone();
		train_set_up();
		ml_perceptron::copy(other, scope);
	}
	gd_net (const gd_net& net, std::string scope);

	virtual ml_perceptron* clone_impl (std::string scope)
	{
		return new gd_net(*this, scope);
	}

private:
	size_t n_input;
	double learning_rate = 0.5; // implement setter
	bool record_training = false;
	// training input
	nnet::placeholder<double>* train_in_ = nullptr;
	nnet::placeholder<double>* expected_out_ = nullptr;
	nnet::placeholder<double>* batch_size = nullptr;
	// training output
	nnet::varptr<double> diff_;
	// training executors
	nnet::group<double>* updates;
	// owns optimizer
	ioptimizer<double>* optimizer_ = nullptr;
};

}

#endif /* gd_net_hpp */
#endif