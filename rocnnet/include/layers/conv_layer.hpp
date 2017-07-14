//
//  conv_layer.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-07-13.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#include <string>
#include <vector>

#include "graph/leaf/variable.hpp"
#include "graph/operations/operations.hpp"

#include "layers/ilayer.hpp"

#pragma once
#ifndef ROCNNET_CONV_LAYER_HPP
#define ROCNNET_CONV_LAYER_HPP

namespace rocnnet
{

class conv_layer : public ilayer
{
public:
	conv_layer (std::pair<size_t,size_t> filter_hw,
		size_t in_ncol, size_t out_ncol, std::string scope = "");

	virtual ~conv_layer (void);

	conv_layer* clone (std::string scope = "") const;

	conv_layer* move (void);

	conv_layer& operator = (const conv_layer& other);

	conv_layer& operator = (conv_layer&& other);

	// input are expected to have shape n_input by batch_size
	// outputs are expected to have shape n_output by batch_size
	virtual nnet::varptr<double> operator () (nnet::inode<double>* input);

	virtual std::vector<nnet::variable<double>*> get_variables (void) const;

protected:
	conv_layer (const conv_layer& other, std::string scope="");

	conv_layer (conv_layer&& other);

	virtual ilayer* clone_impl (std::string& scope) const;

	virtual ilayer* move_impl (void);

private:
	void copy_helper (const conv_layer& other);

	void move_helper (conv_layer&& other);

	void clean_up (void);

	nnet::variable<double>* weight_;

	// bias has shape <out_ncol>
	nnet::variable<double>* bias_;
};

}

#endif /* ROCNNET_CONV_LAYER_HPP */
