//
//  mlp.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "mlp.hpp"

#ifdef ROCNNET_MLP_HPP

namespace rocnnet
{

ml_perceptron::ml_perceptron (
	size_t n_input,
	std::vector<IN_PAIR> hiddens,
    std::string scope) :
n_input_(n_input),
n_output_(hiddens.back().first),
scope_(scope)
{
	size_t level = 0;
	size_t n_output;
	perceptron* percept;
	for (IN_PAIR ip : hiddens)
	{
		n_output = ip.first;
		percept = new perceptron(n_input, n_output,
			nnutils::formatter() << scope_ << ":hidden_" << level++);
		layers_.push_back(HID_PAIR(percept, ip.second));
		n_input = n_output;
	}
}

ml_perceptron::~ml_perceptron (void)
{
	for (HID_PAIR hp : layers_)
	{
		// delete perceptrons to kill the graph
		delete hp.first;
	}
}

ml_perceptron* ml_perceptron::clone (std::string scope)
{
    return clone_impl(scope);
}

ml_perceptron* ml_perceptron::move (std::string scope)
{
	return move_impl(scope);
}

ml_perceptron& ml_perceptron::operator = (const ml_perceptron& other)
{
	if (&other != this)
	{
		copy_helper(other, scope_);
	}
	return *this;
}

ml_perceptron& ml_perceptron::operator = (ml_perceptron&& other)
{
	if (&other != this)
	{
		move_helper(std::move(other), scope_);
	}
	return *this;
}

void ml_perceptron::initialize (std::string serialname, std::string readscope)
{
	if (readscope.empty()) readscope = scope_;
	std::vector<nnet::inode<double>*> vars;
	for (HID_PAIR hp : layers_)
	{
		WB_PAIR wb = hp.first->get_variables();
		vars.push_back(wb.first);
		vars.push_back(wb.second);
	}

	if (nnet::read_inorder<double>(vars, readscope, serialname) && vars.empty()) return;

	for (nnet::inode<double>* v : vars)
	{
		static_cast<nnet::variable<double>*>(v)->initialize();
	}
}

// input are expected to have shape n_input by batch_size
// outputs are expected to have shape output by batch_size
nnet::varptr<double> ml_perceptron::operator () (nnet::inode<double>* input)
{
	// sanity check
	nnet::tensorshape in_shape = input->get_shape();
	assert(in_shape.is_compatible_with(std::vector<size_t>{n_input_, 0}));
	// output of one layer's dimensions is expected to be matched by
	// the perceptron of the next layer
	nnet::inode<double>* output = input;
	for (HID_PAIR hp : layers_)
	{
		nnet::inode<double>* hypothesis = (*hp.first)(output);
		output = (hp.second)(hypothesis);
	}
	if (nnet::immutable<double>* out = dynamic_cast<nnet::immutable<double>*>(output))
	{
		solo_merge(out);
		output = out;
	}
	return output;
}

std::vector<WB_PAIR> ml_perceptron::get_variables (void) const
{
	std::vector<WB_PAIR> wb_vec;
	for (HID_PAIR hp : layers_)
	{
		wb_vec.push_back(hp.first->get_variables());
	}
	return wb_vec;
}

bool ml_perceptron::save (std::string fname) const
{
	std::vector<nnet::inode<double>*> vars;
	for (HID_PAIR hp : layers_)
	{
		WB_PAIR wb = hp.first->get_variables();
		vars.push_back(wb.first);
		vars.push_back(wb.second);
	}
	return nnet::write_inorder<double>(vars, scope_, fname);
}

ml_perceptron::ml_perceptron (const ml_perceptron& other, std::string& scope)
{
	copy_helper(other, scope);
}

ml_perceptron::ml_perceptron (ml_perceptron&& other, std::string& scope)
{
	move_helper(std::move(other), scope);
}

ml_perceptron* ml_perceptron::clone_impl (std::string& scope)
{
	return new ml_perceptron (*this, scope);
}

ml_perceptron* ml_perceptron::move_impl (std::string& scope)
{
	return new ml_perceptron (std::move(*this), scope);
}

void ml_perceptron::copy_helper (const ml_perceptron& other, std::string& scope)
{
	n_input_ = other.n_input_;
	n_output_ = other.n_output_;
	for (HID_PAIR hp : layers_)
	{
		delete hp.first;
	}
	layers_.clear();
	if (0 == scope.size())
	{
		scope_ = other.scope_ + "_cpy";
	}
	else
	{
		scope_ = scope;
	}
	size_t level = 0;
	perceptron* percept;
	for (HID_PAIR hp : other.layers_)
	{
		percept = new perceptron(*hp.first,
			nnutils::formatter() << scope_ << ":hiddens" << level++);
		layers_.push_back(HID_PAIR(percept, hp.second));
	}
}

void ml_perceptron::move_helper (ml_perceptron&& other, std::string& scope)
{
	n_input_ = std::move(other.n_input_);
	n_output_ = std::move(other.n_output_);
	for (HID_PAIR hp : layers_)
	{
		delete hp.first;
	}
	layers_.clear();
	if (0 == scope_.size())
	{
		scope_ = std::move(other.scope_);
	}
	else
	{
		scope_ = scope;
	}
	size_t level = 0;
	perceptron* percept;
	for (HID_PAIR hp : other.layers_)
	{
		percept = new perceptron(std::move(*hp.first),
			nnutils::formatter() << scope_ << ":hiddens" << level++);
		layers_.push_back(HID_PAIR(percept, hp.second));
	}
}

}

#endif