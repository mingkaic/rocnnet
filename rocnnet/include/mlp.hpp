//
//  mlp.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "utils/futils.hpp"
#include "perceptron.hpp"

#pragma once
#ifndef ROCNNET_MLP_HPP
#define ROCNNET_MLP_HPP

namespace rocnnet
{

using VAR_FUNC = std::function<nnet::varptr<double>(nnet::inode<double>*)>;
using IN_PAIR = std::pair<size_t, VAR_FUNC>;
using HID_PAIR = std::pair<perceptron*, VAR_FUNC>;

class ml_perceptron
{
public:
	// trust that passed in operations are unconnected
	ml_perceptron (size_t n_input, std::vector<IN_PAIR> hiddens,
		std::string scope = "MLP");

	virtual ~ml_perceptron (void);

	ml_perceptron* clone (std::string scope = "");

	ml_perceptron* move (std::string scope = "");

	ml_perceptron& operator = (const ml_perceptron& other);

	ml_perceptron& operator = (ml_perceptron&& other);

	void initialize (void);

	// PLACEHOLDER CONNECTION
	// input are expected to have shape n_input by batch_size
	// outputs are expected to have shape output by batch_size
	nnet::varptr<double> operator () (nnet::inode<double>* input);

	std::vector<WB_PAIR> get_variables (void) const;

protected:
	ml_perceptron (const ml_perceptron& other, std::string& scope);

	ml_perceptron (ml_perceptron&& other, std::string& scope);

	virtual ml_perceptron* clone_impl (std::string& scope);

	virtual ml_perceptron* move_impl (std::string& scope);

	std::vector<HID_PAIR> layers;

private:
	void copy_helper (const ml_perceptron& other, std::string& scope);

	void move_helper (ml_perceptron&& other, std::string& scope);

	size_t n_input_;

	std::string scope_;
};

}

#endif /* ROCNNET_MLP_HPP */
