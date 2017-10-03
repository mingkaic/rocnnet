//
//  rbm.hpp
//  cnnet
//
//	Implements Restricted Boltzmann Machine
//	Setup cost and training follows implemented from http://deeplearning.net/tutorial/code/rbm.py
//
//  Created by Mingkai Chen on 2017-07-17.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#include "graph/connector/immutable/generator.hpp"
#include "utils/gd_utils.hpp"

#include "compounds/icompound.hpp"

#pragma once
#ifndef ROCNNET_RBM_HPP
#define ROCNNET_RBM_HPP

namespace rocnnet
{

using generators_t = std::vector<nnet::generator<double>*>;
using update_cost_t = std::pair<nnet::variable_updater<double>, nnet::varptr<double> >;

// todo: toggle activation, sigmoid (current) and ReLU
class rbm : public icompound
{
public:
	// trust that passed in operations are unconnected
	rbm (size_t n_input, size_t n_hidden,
		std::string scope = "RBM");

	virtual ~rbm (void);

	rbm* clone (std::string scope = "") const;

	rbm* move (void);

	rbm& operator = (const rbm& other);

	rbm& operator = (rbm&& other);

	// PLACEHOLDER CONNECTION
	// accept input of shape <n_input, n_batch>
	// output of shape <n_hidden, n_batch>
	virtual nnet::varptr<double> prop_up (nnet::inode<double>* input);

	// accept input of shape <n_hidden, n_batch>
	// output shape of <n_input, n_batch>
	nnet::varptr<double> prop_down (nnet::inode<double>* hidden);

	// recreate input using hidden distribution,
	// output shape of input.get_shape()
	nnet::varptr<double> reconstruct_visible (nnet::inode<double>* input);

	nnet::varptr<double> reconstruct_hidden (nnet::inode<double>* hidden);

	// input a 2-D vector of shape <n_input, n_batch>
	update_cost_t train (
		nnet::inode<double>* input,
		nnet::variable<double>* persistent = nullptr,
		double learning_rate = 1e-3,
		size_t n_cont_div = 1);

	virtual std::vector<nnet::variable<double>*> get_variables (void) const;

	virtual size_t get_ninput (void) const { return n_input_; }

	virtual size_t get_noutput (void) const { return n_hidden_; }

protected:
	rbm (const rbm& other, std::string& scope);

	rbm (rbm&& other);

	virtual ilayer* clone_impl (std::string& scope) const;

	virtual ilayer* move_impl (void);

private:
	void copy_helper (const rbm& other);

	void move_helper (rbm&& other);

	void clean_up (void);

	nnet::varptr<double> free_energy (nnet::varptr<double> sample);

	// COST CALCULATIONS
	nnet::varptr<double> get_pseudo_likelihood_cost (nnet::inode<double>* input);

	nnet::varptr<double> get_reconstruction_cost (nnet::inode<double>* input, nnet::varptr<double>& visible_dist);

	size_t n_input_;

	size_t n_hidden_;

	nnet::variable<double>* weight_ = nullptr;

	nnet::variable<double>* hbias_ = nullptr;

	nnet::variable<double>* vbias_ = nullptr;
};

}

#endif /* ROCNNET_RBM_HPP */
