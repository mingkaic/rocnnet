//
// Created by Mingkai Chen on 2017-07-26.
//

#include "models/db_net.hpp"

#ifdef ROCNNET_DB_NET_HPP

namespace rocnnet
{

db_net::db_net (size_t n_input, std::vector<size_t> hiddens, std::string scope) :
icompound(scope),
n_input_(n_input)
{
	assert(false == hiddens.empty());
	size_t level = 0;
	rbm* layer;
	auto it = hiddens.begin();
	if (hiddens.size() > 1)
	{
		for (auto et = hiddens.end(); it+1 != et; it++)
		{
			n_output_ = *it;
			layer = new rbm(n_input, n_output_,
				nnutils::formatter() << scope_ << ":rbm_" << level++);
			layers_.push_back(layer);
			n_input = n_output_;
		}
	}

	n_output_ = *it;
	log_layer_ = new fc_layer({ n_input }, n_output_, scope_ + ":logres");
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
		copy_helper(other);
	}
	return *this;
}

db_net& db_net::operator = (db_net&& other)
{
	if (&other != this)
	{
		ilayer::operator = (std::move(other));
		move_helper(std::move(other));
	}
	return *this;
}

nnet::varptr<double> db_net::operator () (nnet::placeholder<double>& input)
{
	// sanity check
	nnet::tensorshape in_shape = input.get_shape();
	assert(in_shape.is_compatible_with(std::vector<size_t>{n_input_, 0}));
	nnet::inode<double>* output = &input;
	for (rbm* h : layers_)
	{
		output = h->prop_up(output);
	}
	return nnet::softmax((*log_layer_)({ output }));
}

pretrain_t db_net::pretraining_functions (
	nnet::placeholder<double>& input,
	double learning_rate, size_t n_cont_div)
{
	pretrain_t pt_updates;
	for (rbm* rlayer : layers_)
	{
		pt_updates.push_back(rlayer->train(input, nullptr, learning_rate, n_cont_div));
	}
	return pt_updates;
}

update_cost_t db_net::build_finetune_functions (
	nnet::placeholder<double>& train_in,
	nnet::placeholder<double>& train_out,
	double learning_rate)
{
	nnet::varptr<double> out_dist = (*this)(train_in);
	nnet::varptr<double> finetune_cost = - nnet::reduce_mean(nnet::log(out_dist));
	nnet::iconnector<double>* ft_cost_icon = static_cast<nnet::iconnector<double>*>(finetune_cost.get());

	nnet::varptr<double> prediction = nnet::arg_max(out_dist, 1);

	nnet::varptr<double> error = nnet::reduce_mean(nnet::neq(prediction, nnet::varptr<double>(&train_out)));

	std::vector<nnet::variable<double>*> gparams = this->get_variables();
	std::vector<nnet::variable_updater<double> > uvec;
	for (nnet::variable<double>* gp : gparams)
	{
		uvec.push_back(gp->assign_sub(learning_rate * finetune_cost->derive(gp)));
	}

	return { [uvec, ft_cost_icon](bool)
	{
		ft_cost_icon->freeze_status(true); // freeze
		for (nnet::variable_updater<double> trainer : uvec)
		{
		 trainer(true);
		}
		ft_cost_icon->freeze_status(false); // update again
	}, error };
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
	clean_up();
	n_input_ = other.n_input_;
	n_output_ = other.n_output_;
	for (size_t i = 0, n = other.layers_.size(); i < n; i++)
	{
		rbm* rlayer = other.layers_[i]->clone(
			nnutils::formatter() << scope_ << ":rbm_" << i);
		layers_.push_back(rlayer);
	}
}

void db_net::move_helper (db_net&& other)
{
	clean_up();
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
	if (log_layer_)
	{
		delete log_layer_;
		log_layer_ = nullptr;
	}
	layers_.clear();
}

}

#endif
