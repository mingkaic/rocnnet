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

rbm::rbm (size_t n_input, size_t n_hidden, std::string scope) :
	icompound(scope),
	n_input_(n_input)
{
	nnet::const_init<double> zinit(0);

	double limits = 4 * std::sqrt(6.0 / (n_hidden + n_input));
	nnet::rand_uniform<double> rinit(-limits, limits);

	weight_ = new nnet::variable<double>(std::vector<size_t>{n_hidden, n_input},
		rinit, nnutils::formatter() << scope << "_weights");
	hbias_ = new nnet::variable<double>(std::vector<size_t>{n_hidden},
		zinit, nnutils::formatter() << scope << "_hidden_bias");
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

nnet::varptr<double> rbm::prop_up (nnet::inode<double>* input)
{
	// prop forward
	// weight is <n_hidden, n_input>
	// in is <n_input, ?>
	// out = in @ weight, so out is <n_hidden, ?>
	nnet::varptr<double> weighed = nnet::matmul<double>(input, weight_);
	nnet::varptr<double> pre_nl = nnet::add_axial_b(weighed, nnet::varptr<double>(hbias_), 1);
	return nnet::sigmoid<double>(pre_nl);
}

nnet::varptr<double> rbm::prop_down (nnet::inode<double>* hidden)
{
	// weight is <n_hidden, n_input>
	// in is <n_hidden, ?>
	// out = in @ weight.T, so out is <n_input, ?>
	nnet::varptr<double> weighed = nnet::matmul<double>(hidden, weight_, false, true);
	nnet::varptr<double> pre_nl = nnet::add_axial_b(weighed, nnet::varptr<double>(vbias_), 1);
	return nnet::sigmoid<double>(pre_nl);
}

nnet::varptr<double> rbm::reconstruct_visible (nnet::inode<double>* input)
{
	nnet::varptr<double> hidden_dist = this->prop_up(input);
	nnet::varptr<double> hidden_sample = nnet::binomial_sample(1.0, hidden_dist);
	return this->prop_down(hidden_sample);
}

nnet::varptr<double> rbm::reconstruct_hidden (nnet::inode<double>* hidden)
{
	nnet::varptr<double> visible_dist = this->prop_down(hidden);
	nnet::varptr<double> visible_sample = nnet::binomial_sample(1.0, visible_dist);
	return this->prop_up(visible_sample);
}

std::pair<nnet::variable_updater<double>, nnet::varptr<double> > rbm::train (
	nnet::placeholder<double>& input,
	nnet::variable<double>* persistent,
	double learning_rate,
	size_t n_cont_div)
{
	nnet::inode<double>* chain_it;
	// if persistent not available use Contrastive Divergence (CD)
	if (nullptr == persistent)
	{
		nnet::varptr<double> hidden_dist = this->prop_up(&input);
		chain_it = nnet::binomial_sample(1.0, hidden_dist);
	}
	// otherwise use Persistent CD (initialize from the old state of the chain)
	else
	{
		chain_it = persistent;
	}

	// this series of chaining is implemented from http://deeplearning.net/tutorial/code/rbm.py
	nnet::varptr<double> final_presig_vis;
	nnet::varptr<double> final_visible_dist;
	for (size_t i = 0; i < n_cont_div; i++)
	{
		nnet::varptr<double> hidden_dist = this->reconstruct_hidden(chain_it);

		// use operational optimization to recover presig and vis nodes
		nnet::varptr<double> weighed = nnet::matmul<double>(chain_it, weight_, false, true);
		final_presig_vis =  nnet::add_axial_b(weighed, nnet::varptr<double>(vbias_), 1);
		final_visible_dist = nnet::sigmoid<double>(final_presig_vis);

		chain_it = nnet::binomial_sample(1.0, hidden_dist);
	}
	nnet::varptr<double> final_visible_sample = nnet::binomial_sample(1.0, final_visible_dist);
	// chain_end is treated like a constant
	nnet::varptr<double> chain_end = nnet::as_constant(final_visible_sample);

	nnet::varptr<double> cost = nnet::reduce_mean(this->free_energy(&input)) - nnet::reduce_mean(this->free_energy(chain_end));
	nnet::iconnector<double>* cost_icon = static_cast<nnet::iconnector<double>*>(cost.get());

	nnet::varptr<double> dW = cost->derive(weight_);
	nnet::varptr<double> dhb = cost->derive(hbias_);
	nnet::varptr<double> dvb = cost->derive(vbias_);

	nnet::updates_t uvec;
	uvec.push_back(weight_->assign_sub(learning_rate * dW));
	uvec.push_back(hbias_->assign_sub(learning_rate * dhb));
	uvec.push_back(vbias_->assign_sub(learning_rate * dvb));

	nnet::varptr<double> monitoring_cost;
	if (nullptr == persistent)
	{
		// reconstruction cost
		monitoring_cost = this->get_reconstruction_cost(final_presig_vis);
	}
	else
	{
		// pseudo-likelihood
		uvec.push_back(persistent->assign(chain_it));
		monitoring_cost = this->get_pseudo_likelihood_cost(input);
	}

	return { [uvec, cost_icon](bool)
	{
		cost_icon->freeze_status(true); // freeze
		for (nnet::variable_updater<double> trainer : uvec)
		{
			trainer(true);
		}
		cost_icon->freeze_status(false); // update again
	}, monitoring_cost };
}

std::vector<nnet::variable<double>*> rbm::get_variables (void) const
{
	std::vector<nnet::variable<double>*> vars;
	vars.push_back(weight_);
	vars.push_back(hbias_);
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
	weight_ = other.weight_->clone();
	hbias_ = other.hbias_->clone();
	vbias_ = other.vbias_->clone();
}

void rbm::move_helper (rbm&& other)
{
	n_input_ = std::move(other.n_input_);
	weight_ = other.weight_->move();
	hbias_ = other.hbias_->move();
	vbias_ = other.vbias_->move();
}

void rbm::clean_up (void)
{
	if (weight_) delete weight_;
	if (hbias_) delete hbias_;
	if (vbias_) delete vbias_;

	weight_ = nullptr;
	hbias_ = nullptr;
	vbias_ = nullptr;
}

nnet::varptr<double> rbm::free_energy (nnet::varptr<double> sample)
{
	nnet::varptr<double> vbias_term = nnet::matmul(sample, nnet::varptr<double>(vbias_), false, true);
	// <x, y> @ <z, x> + z
	nnet::varptr<double> weighed = nnet::matmul<double>(sample, nnet::varptr<double>(weight_));
	nnet::varptr<double> wx_b = nnet::add_axial_b(weighed, nnet::varptr<double>(hbias_), 1);
	nnet::varptr<double> hidden_term = nnet::reduce_sum<double>(nnet::log(1.0 + nnet::exp(wx_b)), 0);
	return -(hidden_term + vbias_term);
}

// implementation taken from http://deeplearning.net/tutorial/rbm.html
nnet::varptr<double> rbm::get_pseudo_likelihood_cost (nnet::placeholder<double>& input)
{
	// zeros everywhere except for x-axis = x_idx (x is the first dimension)
	nnet::varptr<double> one_i = nnet::const_axis<double>(0, 0, 1, input.get_shape());

	nnet::varptr<double> xi = nnet::round(nnet::varptr<double>(&input)); // xi = [0|1...]
	nnet::varptr<double> xi_flip = one_i - xi;

	nnet::varptr<double> fe_xi = free_energy(xi);
	nnet::varptr<double> fe_xi_flip = free_energy(xi_flip);

	return nnet::reduce_mean((double) n_input_ * nnet::log(nnet::sigmoid(fe_xi_flip - fe_xi)));
}

nnet::varptr<double> rbm::get_reconstruction_cost (nnet::varptr<double>& pre_sig_back)
{
	return nullptr;
}

}

#endif
