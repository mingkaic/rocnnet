//
//  mlp.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "compound/mlp.hpp"

#ifdef ROCNNET_MLP_HPP

namespace rocnnet
{

mlp::mlp (size_t n_input, std::vector<IN_PAIR> hiddens, std::string scope) :
	ilayer(scope),
	n_input_(n_input),
	n_output_(hiddens.back().first)
{
	size_t level = 0;
	size_t n_output;
	fc_layer* percept;
	for (IN_PAIR ip : hiddens)
	{
		n_output = ip.first;
		percept = new fc_layer({n_input}, n_output,
			nnutils::formatter() << scope_ << ":hidden_" << level++);
		layers_.push_back(HID_PAIR(percept, ip.second));
		n_input = n_output;
	}
}

mlp::~mlp (void)
{
	clean_up();
}

mlp* mlp::clone (std::string scope) const
{
	return static_cast<mlp*>(this->clone_impl(scope));
}

mlp* mlp::move (void)
{
	return static_cast<mlp*>(this->move_impl());
}

mlp& mlp::operator = (const mlp& other)
{
	if (&other != this)
	{
		clean_up();
		copy_helper(other);
	}
	return *this;
}

mlp& mlp::operator = (mlp&& other)
{
	if (&other != this)
	{
		clean_up();
		move_helper(std::move(other));
	}
	return *this;
}

void mlp::initialize (std::string serialname, std::string readscope)
{
	if (readscope.empty()) readscope = scope_;
	std::vector<nnet::inode<double>*> vars;
	for (HID_PAIR hp : layers_)
	{
		std::vector<nnet::variable<double>*> temp = hp.first->get_variables();
		vars.insert(vars.end(), temp.begin(), temp.end());
	}

	if (nnet::read_inorder<double>(vars, readscope, serialname) && vars.empty()) return;

	for (nnet::inode<double>* v : vars)
	{
		static_cast<nnet::variable<double>*>(v)->initialize();
	}
}

// input are expected to have shape n_input by batch_size
// outputs are expected to have shape output by batch_size
nnet::varptr<double> mlp::operator () (nnet::inode<double>* input)
{
	// sanity check
	nnet::tensorshape in_shape = input->get_shape();
	assert(in_shape.is_compatible_with(std::vector<size_t>{n_input_, 0}));
	// output of one fc_layer's dimensions is expected to be matched by
	// the perceptron of the next fc_layer
	nnet::inode<double>* output = input;
	for (HID_PAIR hp : layers_)
	{
		nnet::inode<double>* hypothesis = (*hp.first)(output);
		output = (hp.second)(hypothesis);
	}
	return output;
}

std::vector<nnet::variable<double>*> mlp::get_variables (void) const
{
	std::vector<nnet::variable<double>*> vars;
	for (HID_PAIR hp : layers_)
	{
		std::vector<nnet::variable<double>*> temp = hp.first->get_variables();
		vars.insert(vars.end(), temp.begin(), temp.end());
	}
	return vars;
}

bool mlp::save (std::string fname) const
{
	std::vector<nnet::inode<double>*> vars;
	for (HID_PAIR hp : layers_)
	{
		std::vector<nnet::variable<double>*> temp = hp.first->get_variables();
		vars.insert(vars.end(), temp.begin(), temp.end());
	}
	return nnet::write_inorder<double>(vars, scope_, fname);
}

mlp::mlp (const mlp& other, std::string& scope) :
	ilayer(other, scope)
{
	copy_helper(other);
}

mlp::mlp (mlp&& other) :
	ilayer(std::move(other))
{
	move_helper(std::move(other));
}

ilayer* mlp::clone_impl (std::string& scope) const
{
	return new mlp (*this, scope);
}

ilayer* mlp::move_impl (void)
{
	return new mlp (std::move(*this));
}

void mlp::copy_helper (const mlp& other)
{
	n_input_ = other.n_input_;
	n_output_ = other.n_output_;
	for (size_t i = 0, n = other.layers_.size(); i < n; i++)
	{
		fc_layer* percept = other.layers_[i].first->clone(
			nnutils::formatter() << scope_ << ":hiddens" << i);
		layers_.push_back(HID_PAIR(percept, other.layers_[i].second));
	}
}

void mlp::move_helper (mlp&& other)
{
	n_input_ = std::move(other.n_input_);
	n_output_ = std::move(other.n_output_);
	layers_ = std::move(other.layers_);
}

void mlp::clean_up (void)
{
	for (HID_PAIR hp : layers_)
	{
		delete hp.first;
	}
	layers_.clear();
}

}

#endif