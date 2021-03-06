//
//  mlp.hpp
//  cnnet
//
//	Implements Multilayer Perceptron
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "compounds/icompound.hpp"

#pragma once
#ifndef ROCNNET_MLP_HPP
#define ROCNNET_MLP_HPP

namespace rocnnet
{

using FC_PAIR = std::pair<fc_layer*, VAR_FUNC>;

class mlp : public icompound
{
public:
	// trust that passed in operations are unconnected
	mlp (size_t n_input, std::vector<IN_PAIR> hiddens,
		std::string scope = "MLP");

	virtual ~mlp (void);

	mlp* clone (std::string scope = "") const;

	mlp* move (void);

	mlp& operator = (const mlp& other);

	mlp& operator = (mlp&& other);

	// PLACEHOLDER CONNECTION
	// input are expected to have shape n_input by batch_size
	// outputs are expected to have shape output by batch_size
	virtual nnet::varptr<double> prop_up (nnet::inode<double>* input);

	virtual std::vector<nnet::variable<double>*> get_variables (void) const;

	virtual size_t get_ninput (void) const { return n_input_; }

	virtual size_t get_noutput (void) const { return n_output_; }

protected:
	mlp (const mlp& other, std::string& scope);

	mlp (mlp&& other);

	virtual ilayer* clone_impl (std::string& scope) const;

	virtual ilayer* move_impl (void);

private:
	void copy_helper (const mlp& other);

	void move_helper (mlp&& other);

	void clean_up (void);

	size_t n_input_;

	size_t n_output_;

	std::vector<FC_PAIR> layers_;
};

}

#endif /* ROCNNET_MLP_HPP */
