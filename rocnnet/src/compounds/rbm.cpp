//
//  rbm.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-07-17.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#include "compounds/rbm.hpp"
#include "graph/connector/immutable/generator.hpp"

#ifdef ROCNNET_RBM_HPP

namespace rocnnet
{

rbm::rbm (size_t n_input, IN_PAIR hidden_info, double learning_rate, rbm_param param, std::string scope) :
	icompound(scope),
	n_input_(n_input),
	params_(param)
{
	bgd_.set_learning_rate(learning_rate);
	nnet::const_init<double> zinit(0);
	hidden_ = { new fc_layer({n_input}, hidden_info.first), hidden_info.second };
	vbias_ = new nnet::variable<double>(std::vector<size_t>{n_input},
		zinit, nnutils::formatter() << scope << "_visible_bias");
}

rbm::~rbm (void)
{
	clean_up();
}

rbm* rbm::clone (std::string scope) const
{
	return static_cast<rbm*>(this->clone_impl(scope));
}

rbm* rbm::move (void)
{
	return static_cast<rbm*>(this->move_impl());
}

rbm& rbm::operator = (const rbm& other)
{
	if (this == &other)
	{
		ilayer::operator = (other);
		clean_up();
		copy_helper(other);
	}
	return *this;
}

rbm& rbm::operator = (rbm&& other)
{
	if (this == &other)
	{
		ilayer::operator = (std::move(other));
		clean_up();
		move_helper(std::move(other));
	}
	return *this;
}

nnet::varptr<double> rbm::operator () (nnet::inode<double>* input)
{
	// prop forward
	return (hidden_.second)((*hidden_.first)({input}));
}

nnet::varptr<double> rbm::back (nnet::inode<double>* hidden)
{
	nnet::varptr<double> weight = hidden_.first->get_variables()[0];
	nnet::varptr<double> weighed = nnet::matmul<double>(hidden, weight, false, true);
	nnet::varptr<double> pre_nl = nnet::add_axial_b(weighed, nnet::varptr<double>(vbias_), 1);
	return (hidden_.second)(pre_nl);
}

void rbm::train (nnet::inode<double>* input)
{
	// contrastive divergence
	nnet::inode<double>* v0;
	nnet::inode<double>* vt;
	v0 = vt = input;

	nnet::rand_uniform<double> rinit(0, 1);
	// Sampling
//	for (size_t i = 0; i < params_.n_contrastive_divergence_; i++)
//	{
//		nnet::varptr<double> hidden_sample = this->back(vt);
//		nnet::varptr<double> sample = nnet::generator<double>::get(hidden_sample, rinit);
//		nnet::varptr<double> ht = nnet::conditional<double>(sample, hidden_sample, [](double s, double hs) { return s < hs; });
//		vt = rbm::operator () (ht);
//	}
}

std::vector<nnet::variable<double>*> rbm::get_variables (void) const
{
	std::vector<nnet::variable<double>*> vars = hidden_.first->get_variables();
	vars.push_back(vbias_);
	return vars;
}

rbm::rbm (const rbm& other, std::string& scope) :
	icompound(other, scope)
{
	copy_helper(other);
}

rbm::rbm (rbm&& other) :
	icompound(std::move(other))
{
	move_helper(std::move(other));
}

ilayer* rbm::clone_impl (std::string& scope) const
{
	return new rbm (*this, scope);
}

ilayer* rbm::move_impl (void)
{
	return new rbm (std::move(*this));
}

void rbm::copy_helper (const rbm& other)
{
	n_input_ = other.n_input_;
	hidden_ = { other.hidden_.first->clone(), other.hidden_.second };
	vbias_ = other.vbias_->clone();
}

void rbm::move_helper (rbm&& other)
{
	n_input_ = std::move(other.n_input_);
	hidden_ = std::move(other.hidden_);
	vbias_ = other.vbias_->move();
}

void rbm::clean_up (void)
{
	delete hidden_.first;
	if (vbias_) delete vbias_;

	hidden_.first = nullptr;
	vbias_ = nullptr;
}

}

#endif
