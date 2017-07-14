//
//  fc_layer.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "layers/fc_layer.hpp"

#ifdef ROCNNET_FC_LAYER_HPP

namespace rocnnet
{

fc_layer::fc_layer (std::vector<size_t> n_inputs,
	size_t n_output, std::string scope) : ilayer(scope)
{
	nnet::const_init<double> zinit(0);

	for (size_t i = 0, n = n_inputs.size(); i < n; i++)
	{
		nnet::rand_uniform<double> rinit(-1 / std::sqrt(n_inputs[i]), 1 / std::sqrt(n_inputs[i]));
		nnet::variable<double>* weight = new nnet::variable<double>(std::vector<size_t>{n_output, n_inputs[i]},
		   rinit, nnutils::formatter() << scope << "_weights_" << i);
		nnet::variable<double>* bias = new nnet::variable<double>(std::vector<size_t>{n_output},
		   zinit, nnutils::formatter() << scope << "_bias_" << i);
		weights_n_bias_.push_back({weight, bias});
	}
}

fc_layer::~fc_layer (void)
{
	clean_up();
}

fc_layer* fc_layer::clone (std::string scope) const
{
	return static_cast<fc_layer*>(this->clone_impl(scope));
}

fc_layer* fc_layer::move (void)
{
	return static_cast<fc_layer*>(this->move_impl());
}

fc_layer& fc_layer::operator = (const fc_layer& other)
{
	if (&other != this)
	{
		clean_up();
		this->copy_helper(other);
	}
	return *this;
}

fc_layer& fc_layer::operator = (fc_layer&& other)
{
	if (&other != this)
	{
		clean_up();
		this->move_helper(std::move(other));
	}
	return *this;
}

// input are expected to have shape n_input by batch_size
// outputs are expected to have shape n_output by batch_size
nnet::varptr<double> fc_layer::operator () (nnet::inode<double>* input)
{
	// weights are n_output column by n_input rows
	nnet::varptr<double> weighed = nnet::matmul<double>(input, weights_n_bias_[0].first);
	nnet::varptr<double> result = nnet::add_axial_b(weighed,
		nnet::varptr<double>(weights_n_bias_[0].second), 1);
	for (size_t i = 0, n = weights_n_bias_.size(); i < n; i++)
	{
		weighed = nnet::matmul<double>(input, weights_n_bias_[i].first);
		result = result + nnet::add_axial_b(weighed,
			nnet::varptr<double>(weights_n_bias_[i].second), 1);
	}
	return result;
}

std::vector<nnet::variable<double>*> fc_layer::get_variables (void) const
{
	std::vector<nnet::variable<double>*> vars;
	for (WB_PAIR hp : weights_n_bias_)
	{
		vars.push_back(hp.first);
		vars.push_back(hp.second);
	}
	return vars;
}

fc_layer::fc_layer (const fc_layer& other, std::string scope) :
	ilayer(other, scope)
{
	copy_helper(other);
}

fc_layer::fc_layer (fc_layer&& other) :
	ilayer(std::move(other))
{
	move_helper(std::move(other));
}

ilayer* fc_layer::clone_impl (std::string& scope) const
{
	return new fc_layer(*this, scope);
}

ilayer* fc_layer::move_impl (void)
{
	return new fc_layer(std::move(*this));
}

void fc_layer::copy_helper (const fc_layer& other)
{
	for (size_t i = 0, n = other.weights_n_bias_.size(); i < n; i++)
	{
		const WB_PAIR& otherwb = other.weights_n_bias_[i];
		nnet::variable<double>* weight = otherwb.first->clone();
		nnet::variable<double>* bias = otherwb.second->clone();
		weight->set_label(nnutils::formatter() << scope_ << "_weights_" << i);
		bias->set_label(nnutils::formatter() << scope_ << "_bias_" << i);
		weights_n_bias_.push_back({weight, bias});
	}
}

void fc_layer::move_helper (fc_layer&& other)
{
	weights_n_bias_ = std::move(other.weights_n_bias_);
}

void fc_layer::clean_up (void)
{
	for (WB_PAIR wb : weights_n_bias_)
	{
		delete wb.first;
		delete wb.second;
	}
	weights_n_bias_.clear();
}

}

#endif
