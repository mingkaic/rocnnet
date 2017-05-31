//
//  mlp.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "utils/futils.hpp"
#include "perceptron.hpp"
#include "memory/tensor_io.hpp"
#include "graph/optimize/merged_immutable.hpp"

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

	void initialize (std::string serialname = "", std::string readscope = "");

	// PLACEHOLDER CONNECTION
	// input are expected to have shape n_input by batch_size
	// outputs are expected to have shape output by batch_size
	nnet::varptr<double> operator () (nnet::inode<double>* input);

	std::vector<WB_PAIR> get_variables (void) const;

	bool save (std::string fname) const;

	size_t get_ninput (void) const { return n_input_; }

	size_t get_noutput (void) const { return n_output_; }

protected:
	ml_perceptron (const ml_perceptron& other, std::string& scope);

	ml_perceptron (ml_perceptron&& other, std::string& scope);

	virtual ml_perceptron* clone_impl (std::string& scope);

	virtual ml_perceptron* move_impl (std::string& scope);

private:
	void copy_helper (const ml_perceptron& other, std::string& scope);

	void move_helper (ml_perceptron&& other, std::string& scope);

	size_t n_input_;

	size_t n_output_;

	std::string scope_;

	std::vector<HID_PAIR> layers_;
};

}

#endif /* ROCNNET_MLP_HPP */
