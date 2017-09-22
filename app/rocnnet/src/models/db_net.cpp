//
// Created by Mingkai Chen on 2017-07-26.
//

#include "models/db_net.hpp"

#ifdef ROCNNET_DB_NET_HPP

namespace rocnnet
{

db_net::db_net (size_t n_input, std::vector<size_t> hiddens,
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

db_net::~db_net (void)
{
	clean_up();
}

db_net* db_net::clone (std::string scope) const
{
	return static_cast<db_net*>(this->clone_impl(scope));
}

db_net* db_net::move (void)
{
	return static_cast<db_net*>(this->move_impl());
}

db_net& db_net::operator = (const db_net& other)
{
	if (&other != this)
	{
		ilayer::operator = (other);
		clean_up();
		copy_helper(other);
	}
	return *this;
}

db_net& db_net::operator = (db_net&& other)
{
	if (&other != this)
	{
		ilayer::operator = (std::move(other));
		clean_up();
		move_helper(std::move(other));
	}
	return *this;
}

nnet::varptr<double> db_net::operator () (nnet::inode<double>* input)
{
	// sanity check
	nnet::tensorshape in_shape = input->get_shape();
	assert(in_shape.is_compatible_with(std::vector<size_t>{n_input_, 0}));
	nnet::inode<double>* output = input;
	for (rbm* h : layers_)
	{
		output = h->prop_up(output);
	}
	return output;
}

std::vector<nnet::variable<double>*> db_net::get_variables (void) const
{
	std::vector<nnet::variable<double>*> vars;
	for (rbm* h : layers_)
	{
		std::vector<nnet::variable<double>*> temp = h->get_variables();
		vars.insert(vars.end(), temp.begin(), temp.end());
	}
	return vars;
}

size_t db_net::get_ninput (void) const { return n_input_; }

size_t db_net::get_noutput (void) const { return n_output_; }

db_net::db_net (const db_net& other, std::string& scope) :
	icompound(other, scope)
{
	copy_helper(other);
}

db_net::db_net (db_net&& other) :
	icompound(std::move(other))
{
	move_helper(std::move(other));
}

ilayer* db_net::clone_impl (std::string& scope) const
{
	return new db_net(*this, scope);
}

ilayer* db_net::move_impl (void)
{
	return new db_net(std::move(*this));
}

void db_net::copy_helper (const db_net& other)
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

void db_net::move_helper (db_net&& other)
{
	n_input_ = std::move(other.n_input_);
	n_output_ = std::move(other.n_output_);
	layers_ = std::move(other.layers_);
}

void db_net::clean_up (void)
{
	for (rbm* h : layers_)
	{
		delete h;
	}
	layers_.clear();
}

}

#endif
