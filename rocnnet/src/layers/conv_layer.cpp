//
//  conv_layer.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-07-13.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "layers/conv_layer.hpp"

#ifdef ROCNNET_CONV_LAYER_HPP

namespace rocnnet
{

conv_layer::conv_layer (std::pair<size_t,size_t> filter_hw,
	size_t in_ncol, size_t out_ncol, std::string scope) : ilayer(scope)
{
	size_t input_size = filter_hw.first * filter_hw.second * in_ncol;
	nnet::const_init<double> zinit(0);
	nnet::rand_uniform<double> rinit(-1 / std::sqrt(input_size), 1 / std::sqrt(input_size));

	weight_ = new nnet::variable<double>(std::vector<size_t>{filter_hw.first, filter_hw.second, in_ncol, out_ncol},
		rinit, scope + "_weight");
	bias_ = new nnet::variable<double>(std::vector<size_t>{out_ncol}, zinit, scope + "_bias");
}

conv_layer::~conv_layer (void)
{
	clean_up();
}

conv_layer* conv_layer::clone (std::string scope) const
{
	return static_cast<conv_layer*>(this->clone_impl(scope));
}

conv_layer* conv_layer::move (void)
{
	return static_cast<conv_layer*>(this->move_impl());
}

conv_layer& conv_layer::operator = (const conv_layer& other)
{
	if (&other != this)
	{
		clean_up();
		this->copy_helper(other);
	}
	return *this;
}

conv_layer& conv_layer::operator = (conv_layer&& other)
{
	if (&other != this)
	{
		clean_up();
		this->move_helper(std::move(other));
	}
	return *this;
}

nnet::varptr<double> conv_layer::operator () (nnet::inode<double>* input)
{
	nnet::varptr<double> weighed = nnet::conv2d<double>(input, weight_);
	return nnet::add_axial_b(weighed, nnet::varptr<double>(bias_), 1);
}

std::vector<nnet::variable<double>*> conv_layer::get_variables (void) const
{
	std::vector<nnet::variable<double>*> vars;
	vars.push_back(weight_);
	vars.push_back(bias_);
	return vars;
}

conv_layer::conv_layer (const conv_layer& other, std::string scope) :
	ilayer(other, scope)
{
	copy_helper(other);
}

conv_layer::conv_layer (conv_layer&& other) :
	ilayer(std::move(other))
{
	move_helper(std::move(other));
}

ilayer* conv_layer::clone_impl (std::string& scope) const
{
	return new conv_layer(*this, scope);
}

ilayer* conv_layer::move_impl (void)
{
	return new conv_layer(std::move(*this));
}

void conv_layer::copy_helper (const conv_layer& other)
{
	weight_ = other.weight_->clone();
	bias_ = other.bias_->clone();
	weight_->set_label(scope_ + "_weight");
	bias_->set_label(scope_ + "_bias");
}

void conv_layer::move_helper (conv_layer&& other)
{
	weight_ = other.weight_->move();
	bias_ = other.bias_->move();
}

void conv_layer::clean_up (void)
{
	delete weight_;
	delete bias_;
}

}

#endif
