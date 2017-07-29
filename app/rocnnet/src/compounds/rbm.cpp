//
//  rbm.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-07-17.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#include "compounds/rbm.hpp"

#ifdef ROCNNET_RBM_HPP

namespace rocnnet
{

rbm::rbm (size_t n_input, IN_PAIR hidden_info, std::string scope) :
	icompound(scope),
	n_input_(n_input)
{
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
	// weight is <n_hidden, n_input>
	// in is <n_input, ?>
	// out = in @ weight, so out is <n_hidden, ?>
	return (hidden_.second)((*hidden_.first)({input}));
}

nnet::varptr<double> rbm::back (nnet::inode<double>* hidden)
{
	// weight is <n_hidden, n_input>
	// in is <n_hidden, ?>
	// out = in @ weight.T, so out is <n_input, ?>
	nnet::varptr<double> weight = hidden_.first->get_variables()[0];
	nnet::varptr<double> weighed = nnet::matmul<double>(hidden, weight, false, true);
	nnet::varptr<double> pre_nl = nnet::add_axial_b(weighed, nnet::varptr<double>(vbias_), 1);
	return (hidden_.second)(pre_nl);
}

nnet::updates_t rbm::train (generators_t& gens,
	nnet::inode<double>* input, double learning_rate, size_t n_cont_div)
{
	// grad approx by contrastive divergence
	nnet::varptr<double> v0;
	nnet::varptr<double> vt;
	v0 = vt = input; // of shape <n_input, n_batch>

	nnet::rand_normal<double> rinit(0);
	// sampling
	for (size_t i = 0; i < n_cont_div; i++)
	{
		nnet::varptr<double> hidden_sample = rbm::operator () (vt); // <n_hidden, n_batch>
		nnet::generator<double>* gen;
		nnet::varptr<double> sample = gen = nnet::generator<double>::get(hidden_sample, rinit);
		gens.push_back(gen);
		nnet::varptr<double> ht = nnet::conditional<double>(sample, hidden_sample,
			[](double s, double hs) { return s < hs; }, "less");
		vt = this->back(ht);  // <n_input, n_batch>
	}

	// compute deltas
	nnet::varptr<double> h0 = rbm::operator () (v0); // of shape <n_hidden, n_batch>
	nnet::varptr<double> hk = rbm::operator () (vt);
	v0 = nnet::reduce_mean(v0, 1); // reduce to shape <n_input>
	vt = nnet::reduce_mean(vt, 1);
	h0 = nnet::reduce_mean(h0, 1); // reduce to shape <n_hidden>
	hk = nnet::reduce_mean(hk, 1);
	nnet::varptr<double> dW = nnet::matmul<double>(h0, v0, true) - nnet::matmul<double>(hk, vt, true); // of shape <n_hidden, n_input>
	nnet::varptr<double> dhb = h0 - hk;
	nnet::varptr<double> dvb = v0 - vt;

	std::vector<nnet::variable<double>*> vars = hidden_.first->get_variables();
	nnet::variable<double>* weight = vars[0];
	nnet::variable<double>* hbias = vars[1];

	nnet::updates_t uvec;
	uvec.push_back(weight->assign_add(learning_rate * dW));
	uvec.push_back(hbias->assign_add(learning_rate * dhb));
	uvec.push_back(vbias_->assign_add(learning_rate * dvb));
	return uvec;
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

void fit (rbm& model, std::vector<double> batch, rbm_param params)
{
	size_t n_input = model.get_ninput();
	assert(0 == batch.size() % n_input);
	size_t n_batch = batch.size() / n_input;
	nnet::placeholder<double> in(std::vector<size_t>{n_input, n_batch}, "rbm_train_in");
	rocnnet::generators_t gens;
	nnet::updates_t trainers = model.train(gens, &in, params.learning_rate_, params.n_cont_div_);
	trainers.push_back([gens](bool)
	{
		for (nnet::generator<double>* gen : gens)
		{
			gen->update({}); // re-randomize
		}
	});
	in = batch;
	for (size_t i = 0; i < params.n_epoch_; i++)
	{
		for (nnet::variable_updater<double>& trainer : trainers)
		{
			trainer(true);
		}
	}
}

}

#endif