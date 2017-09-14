//
//  rbm.hpp
//  cnnet
//
//	Implements Restricted Boltzmann Machine
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
	nnet::varptr<double> operator () (nnet::inode<double>* input);

	// accept input of shape <n_hidden, n_batch>
	// output shape of <n_input, n_batch>
	nnet::varptr<double> back (nnet::inode<double>* hidden);

	// input a 2-D vector of shape <n_input, n_batch>
	nnet::updates_t train (generators_t& gens, nnet::inode<double>* input,
		double learning_rate = 1e-3, size_t n_cont_div = 1);

	nnet::varptr<double> get_pseudo_likelihood_cost (nnet::placeholder<double>& input) const;

	virtual std::vector<nnet::variable<double>*> get_variables (void) const;

	size_t get_ninput (void) const { return n_input_; }

protected:
	rbm (const rbm& other, std::string& scope);

	rbm (rbm&& other);

	virtual ilayer* clone_impl (std::string& scope) const;

	virtual ilayer* move_impl (void);

private:
	void copy_helper (const rbm& other);

	void move_helper (rbm&& other);

	void clean_up (void);

	nnet::varptr<double> free_energy (nnet::varptr<double> sample) const;

	size_t n_input_;

	fc_layer* hidden_ = nullptr;

	nnet::variable<double>* vbias_ = nullptr;
};

}

#endif /* ROCNNET_RBM_HPP */
