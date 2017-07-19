//
//  rbm.hpp
//  cnnet
//
//	Implements Restricted Boltzmann Machine
//
//  Created by Mingkai Chen on 2017-07-17.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#include "compounds/icompound.hpp"
#include "utils/gd_utils.hpp"

#pragma once
#ifndef ROCNNET_RBM_HPP
#define ROCNNET_RBM_HPP

namespace rocnnet
{

struct rbm_param
{
	size_t n_contrastive_divergence_ = 1;
	size_t n_batch_ = 32;
	size_t n_epoch = 10;
	double learning_rate_ = 1e-3;
};

class rbm : public icompound
{
public:
	// trust that passed in operations are unconnected
	rbm (size_t n_input, IN_PAIR hidden_info,
		std::string scope = "RBM");

	virtual ~rbm (void);

	rbm* clone (std::string scope = "") const;

	rbm* move (void);

	rbm& operator = (const rbm& other);

	rbm& operator = (rbm&& other);

	// PLACEHOLDER CONNECTION
	// input are expected to have shape n_input by batch_size
	// outputs are expected to have shape output by batch_size
	nnet::varptr<double> operator () (nnet::inode<double>* input);

	nnet::varptr<double> back (nnet::inode<double>* hidden);

	// input a 2-D vector of shape <n_input, n_batch>
	nnet::updates_t train (nnet::inode<double>* input,
		double learning_rate = 1e-3, size_t n_cont_div = 1);

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

	size_t n_input_;

	HID_PAIR hidden_;

	nnet::variable<double>* vbias_;

	rbm_param params_;
};

}

#endif /* ROCNNET_RBM_HPP */
