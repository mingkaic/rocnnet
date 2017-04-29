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
#include "executor/gd_utils.hpp"

#pragma once
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
	gd_net (size_t n_input, std::vector<IN_PAIR> hiddens,
		nnet::gd_updater<double>& updater, std::string scope = "MLP");

	~gd_net (void);

	gd_net* clone (std::string scope = "");

	gd_net* move (std::string scope = "");

	gd_net& operator = (const gd_net& other);

	gd_net& operator = (gd_net&& other);

	void train (std::vector<double>& train_in, std::vector<double>& expected_out);

protected:
	gd_net (const gd_net& other, std::string& scope);

	gd_net (gd_net&& other, std::string scope);

	virtual ml_perceptron* clone_impl (std::string& scope);

	virtual ml_perceptron* move_impl (std::string& scope);

private:
	void train_setup (void);

	void copy_helper (const gd_net& other);

	void move_helper (gd_net&& other);

	std::vector<nnet::variable_updater<double> > updates_;

	nnet::placeholder<double>* train_in_ = nullptr;

	nnet::placeholder<double>* expected_out_ = nullptr;

	nnet::iconnector<double>* error_ = nullptr;

	nnet::gd_updater<double>* updater_ = nullptr;
};

}

#endif /* gd_net_hpp */