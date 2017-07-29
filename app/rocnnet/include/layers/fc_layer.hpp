//
//  fc_layer.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <string>
#include <vector>

#include "graph/leaf/variable.hpp"
#include "graph/operations/operations.hpp"

#include "layers/ilayer.hpp"

#pragma once
#ifndef ROCNNET_FC_LAYER_HPP
#define ROCNNET_FC_LAYER_HPP

namespace rocnnet
{

class fc_layer : public ilayer
{
public:
	fc_layer (std::vector<size_t> n_inputs,
		size_t n_output, std::string scope = "");

	virtual ~fc_layer (void);

	fc_layer* clone (std::string scope = "") const;

	fc_layer* move (void);

	fc_layer& operator = (const fc_layer& other);

	fc_layer& operator = (fc_layer&& other);

	// input are expected to have shape n_input by batch_size
	// outputs are expected to have shape n_output by batch_size
	nnet::varptr<double> operator () (std::vector<nnet::inode<double>*> inputs);

	virtual std::vector<nnet::variable<double>*> get_variables (void) const;

protected:
	fc_layer (const fc_layer& other, std::string scope);

	fc_layer (fc_layer&& other);

	virtual ilayer* clone_impl (std::string& scope) const;

	virtual ilayer* move_impl (void);

private:
	using WB_PAIR = std::pair<nnet::variable<double>*, nnet::variable<double>*>;

	void copy_helper (const fc_layer& other);

	void move_helper (fc_layer&& other);

	void clean_up (void);

	// weights have shape <output, input>
	// bias has shape <output>
	std::vector<WB_PAIR> weights_n_bias_;
};

}

#endif /* ROCNNET_FC_LAYER_HPP */
