//
//  gd_net.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "compounds/mlp.hpp"
#include "utils/gd_utils.hpp"

#pragma once
#ifndef ROCNNET_GDN_HPP
#define ROCNNET_GDN_HPP

#include <iostream> // for recording
#include <unordered_set>

namespace rocnnet
{

// wrapper for
// gradient descent
class gd_net
{
public:
	gd_net (icompound* brain, nnet::gd_updater& updater,
		std::string scope = "GDN");

	~gd_net (void);

	gd_net (const gd_net& other, std::string scope);

	gd_net (gd_net&& other);

	gd_net& operator = (const gd_net& other);

	gd_net& operator = (gd_net&& other);

	std::vector<double> operator () (std::vector<double>& input);

	void train (std::vector<double>& train_in, std::vector<double>& expected_out);

	void initialize (std::string serialname = "", std::string readscope = "");

	bool save (std::string fname, std::string writescope = "") const;

	// expose error to analyze graph
	const nnet::iconnector<double>* get_error (void) const { return error_; }

private:
	void setup (void);

	void copy_helper (const gd_net& other, std::string scope);

	void move_helper (gd_net&& other);

	void clean_up (void);

	nnet::updates_t updates_;

	icompound* brain_ = nullptr;

	nnet::placeholder<double>* test_in_ = nullptr;

	nnet::varptr<double> test_out_;

	nnet::placeholder<double>* train_in_ = nullptr;

	nnet::placeholder<double>* expected_out_ = nullptr;

	nnet::iconnector<double>* error_ = nullptr;

	nnet::gd_updater* updater_ = nullptr;

	std::string scope_;
};

}

#endif /* ROCNNET_GDN_HPP */