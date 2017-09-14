//
// Created by Mingkai Chen on 2017-07-26.
//

#include "compounds/dbn.hpp"

#ifdef ROCNNET_DBN_HPP

namespace rocnnet
{

dbn::dbn (size_t n_input, std::vector<size_t> hiddens,
	dbn_param train_param, std::string scope) :
icompound(scope),
n_input_(n_input)
{
	size_t level = 0;
	rbm* rlayer;
	for (size_t hid_size : hiddens)
	{
		n_output_ = hid_size;
		rlayer = new rbm(n_input, n_output_,
						 nnutils::formatter() << scope_ << ":hidden_" << level++);
		layers_.push_back(rlayer);
		n_input = n_output_;
	}
}

dbn::~dbn (void)
{
	clean_up();
}

dbn* dbn::clone (std::string scope) const
{
	return static_cast<dbn*>(this->clone_impl(scope));
}

dbn* dbn::move (void)
{
	return static_cast<dbn*>(this->move_impl());
}

dbn& dbn::operator = (const dbn& other)
{
	if (&other != this)
	{
		ilayer::operator = (other);
		clean_up();
		copy_helper(other);
	}
	return *this;
}

dbn& dbn::operator = (dbn&& other)
{
	if (&other != this)
	{
		ilayer::operator = (std::move(other));
		clean_up();
		move_helper(std::move(other));
	}
	return *this;
}

nnet::varptr<double> dbn::operator () (nnet::inode<double>* input)
{
	// sanity check
	nnet::tensorshape in_shape = input->get_shape();
	assert(in_shape.is_compatible_with(std::vector<size_t>{n_input_, 0}));
	nnet::inode<double>* output = input;
	for (rbm* h : layers_)
	{
		output = (*h)(output);
	}
	return output;
}

std::vector<nnet::variable<double>*> dbn::get_variables (void) const
{
	std::vector<nnet::variable<double>*> vars;
	for (rbm* h : layers_)
	{
		std::vector<nnet::variable<double>*> temp = h->get_variables();
		vars.insert(vars.end(), temp.begin(), temp.end());
	}
	return vars;
}

size_t dbn::get_ninput (void) const { return n_input_; }

size_t dbn::get_noutput (void) const { return n_output_; }

dbn::dbn (const dbn& other, std::string& scope) :
	icompound(other, scope)
{
	copy_helper(other);
}

dbn::dbn (dbn&& other) :
	icompound(std::move(other))
{
	move_helper(std::move(other));
}

ilayer* dbn::clone_impl (std::string& scope) const
{
	return new dbn(*this, scope);
}

ilayer* dbn::move_impl (void)
{
	return new dbn(std::move(*this));
}

void dbn::copy_helper (const dbn& other)
{
	n_input_ = other.n_input_;
	n_output_ = other.n_output_;
	for (size_t i = 0, n = other.layers_.size(); i < n; i++)
	{
		rbm* rlayer = other.layers_[i]->clone(
				nnutils::formatter() << scope_ << ":hiddens" << i);
		layers_.push_back(rlayer);
	}
}

void dbn::move_helper (dbn&& other)
{
	n_input_ = std::move(other.n_input_);
	n_output_ = std::move(other.n_output_);
	layers_ = std::move(other.layers_);
}

void dbn::clean_up (void)
{
	for (rbm* h : layers_)
	{
		delete h;
	}
	layers_.clear();
}

}

#endif
