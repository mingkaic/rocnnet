//
// Created by Mingkai Chen on 2017-07-17.
//

#include "compound/icompound.hpp"

#pragma once
#ifndef ROCNNET_RBM_HPP
#define ROCNNET_RBM_HPP

namespace rocnnet
{

class rbm : public icompound
{
public:
	// trust that passed in operations are unconnected
	rbm (size_t n_input, IN_PAIR hiddens,
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

	virtual std::vector<nnet::variable<double>*> get_variables (void) const;

	size_t get_ninput (void) const { return n_input_; }

	size_t get_noutput (void) const { return n_output_; }

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

	size_t n_output_;

	HID_PAIR hiddens_;
};

}

#endif /* ROCNNET_RBM_HPP */
