//
//  layer.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <string>
#include <vector>
#include <random>
#include <list>
#include <stack>

#include "graph/leaf/variable.hpp"
#include "graph/operations/operations.hpp"

#pragma once
#ifndef ROCNNET_PERCEPTRON_HPP
#define ROCNNET_PERCEPTRON_HPP

namespace rocnnet
{

using WB_PAIR = std::pair<nnet::variable<double>*, nnet::variable<double>*>;

// TODO: implement convolution neural net
class perceptron
{
public:
	perceptron (size_t n_input, size_t n_output, std::string scope="");

	virtual ~perceptron (void);

	perceptron (const perceptron& other, std::string scope="");

	perceptron (perceptron&& other);

	perceptron& operator = (const perceptron& other);

	perceptron& operator = (perceptron&& other);

	// input are expected to have shape n_input by batch_size
	// outputs are expected to have shape n_output by batch_size
	nnet::varptr<double> operator () (nnet::inode<double>* input);

	size_t get_n_input (void) const;

	size_t get_n_output (void) const;

	WB_PAIR get_variables (void) const;

protected:
	void copy_helper (const perceptron& other, std::string scope);

	void move_helper (perceptron&& other);

private:
	// metadata
	std::string scope;
	size_t n_input;
	size_t n_output;

	// content
	// weights have shape <output, input>
	// bias has shape <output>
	nnet::variable<double>* weights_ = nullptr;
	nnet::variable<double>* bias_ = nullptr;
};

}

#endif /* ROCNNET_PERCEPTRON_HPP */
