//
//  gd_net.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "compound/mlp.hpp"
#include "utils/gd_utils.hpp"

#pragma once
#ifndef gd_net_hpp
#define gd_net_hpp

#include <iostream> // for recording
#include <unordered_set>

namespace rocnnet
{

// wrapper for
// gradient descent
class gd_net : public mlp
{
public:
	gd_net (size_t n_input, std::vector<IN_PAIR> hiddens,
		nnet::gd_updater& updater, std::string scope = "MLP");

	~gd_net (void);

	gd_net* clone (std::string scope = "") const;

	gd_net* move (void);

	gd_net& operator = (const gd_net& other);

	gd_net& operator = (gd_net&& other);

	void train (std::vector<double>& train_in, std::vector<double>& expected_out);

	// expose error to analyze graph
	const nnet::iconnector<double>* get_error (void) const { return error_; }

protected:
	gd_net (const gd_net& other, std::string& scope);

	gd_net (gd_net&& other);

	virtual ilayer* clone_impl (std::string& scope) const;

	virtual ilayer* move_impl (void);

private:
	void train_setup (void);

	void copy_helper (const gd_net& other);

	void move_helper (gd_net&& other);

	void clean_up (void);

	nnet::updates_t updates_;

	nnet::placeholder<double>* train_in_ = nullptr;

	nnet::placeholder<double>* expected_out_ = nullptr;

	nnet::iconnector<double>* error_ = nullptr;

	nnet::gd_updater* updater_ = nullptr;
};

}

#endif /* gd_net_hpp */