//
// Created by Mingkai Chen on 2017-04-27.
//

#include "utils/gd_utils.hpp"

#ifdef ROCNNET_GD_UTILS_HPP

namespace nnet
{
	
gd_updater::gd_updater (double learning_rate) : learning_rate_(learning_rate) {}

gd_updater::~gd_updater(void) {}

gd_updater* gd_updater::clone (void) const { return clone_impl(); }

gd_updater* gd_updater::move (void) { return move_impl(); }

updates_t gd_updater::calculate (inode<double>* root, grad_process intermediate_process)
{
	std::vector<variable_updater<double> > updates;
	std::unordered_set<ileaf<double>*> leafset = root->get_leaves();
	std::vector<std::pair<inode<double>*,variable<double>*> > gress;
	for (ileaf<double>* l : leafset)
	{
		variable<double>* Wb = dynamic_cast<variable<double>*>(l);
		if (Wb && ignored_.end() == ignored_.find(Wb))
		{
			gress.push_back({root->derive(Wb), Wb});
		}
	}

	for (auto& gpair : gress)
	{
		varptr<double> gres = gpair.first;
		updates.push_back(process_update(gres, gpair.second, intermediate_process));
	}
	return updates;
}

void gd_updater::ignore_subtree (inode<double>* subroot)
{
	std::unordered_set<ileaf<double>*> leafset = subroot->get_leaves();
	for (ileaf<double>* l : leafset)
	{
		if (variable<double>* Wb = dynamic_cast<variable<double>*>(l))
		{
			ignored_.emplace(Wb);
		}
	}
}

void gd_updater::clear_ignore (void)
{
	ignored_.clear();
}

void gd_updater::set_learning_rate (double learning_rate)
{
	learning_rate_ = learning_rate;
}


vgb_updater::vgb_updater (double learning_rate) : gd_updater(learning_rate) {}
		
vgb_updater*  vgb_updater::clone (void) { return static_cast<vgb_updater*>(clone_impl()); }

vgb_updater*  vgb_updater::move (void) { return static_cast<vgb_updater*>(move_impl()); }

gd_updater*  vgb_updater::clone_impl (void) const
{
	return new vgb_updater(*this);
}

gd_updater* vgb_updater::move_impl (void)
{
	return new vgb_updater(std::move(*this));
}

variable_updater<double> vgb_updater::process_update (varptr<double>& gres,
	variable<double>* leaf, grad_process intermediate_process)
{
	// leaf = leaf - learning_rate * gres
	return leaf->assign_sub(intermediate_process(gres, leaf) * learning_rate_);
}


momentum_updater* momentum_updater::clone (void) { return static_cast<momentum_updater*>(clone_impl()); }

momentum_updater* momentum_updater::move (void) { return static_cast<momentum_updater*>(move_impl()); }

gd_updater* momentum_updater::clone_impl (void) const
{
	return new momentum_updater(*this);
}

gd_updater* momentum_updater::move_impl (void)
{
	return new momentum_updater(std::move(*this));
}

variable_updater<double> momentum_updater::process_update (varptr<double>& /*gres*/,
	variable<double>* /*leaf*/, grad_process /*intermediate_process*/)
{
	throw std::bad_function_call();
	return [](bool) {};
}


adadelta_updater* adadelta_updater::clone (void) { return static_cast<adadelta_updater*>(clone_impl()); }

adadelta_updater* adadelta_updater::move (void) { return static_cast<adadelta_updater*>(move_impl()); }

gd_updater* adadelta_updater::clone_impl (void) const
{
	return new adadelta_updater(*this);
}

gd_updater* adadelta_updater::move_impl (void)
{
	return new adadelta_updater(std::move(*this));
}

variable_updater<double> adadelta_updater::process_update (varptr<double>& /*gres*/,
	variable<double>* /*leaf*/, grad_process /*intermediate_process*/)
{
	throw std::bad_function_call();
	return [](bool) {};
}


adagradupdater* adagradupdater::clone (void) { return static_cast<adagradupdater*>(clone_impl()); }

adagradupdater* adagradupdater::move (void) { return static_cast<adagradupdater*>(move_impl()); }

gd_updater* adagradupdater::clone_impl (void) const
{
	return new adagradupdater(*this);
}

gd_updater* adagradupdater::move_impl (void)
{
	return new adagradupdater(std::move(*this));
}

variable_updater<double> adagradupdater::process_update (varptr<double>& /*gres*/,
	variable<double>* /*leaf*/, grad_process /*intermediate_process*/)
{
	throw std::bad_function_call();
	return [](bool) {};
}


rmspropupdater::rmspropupdater (double learning_rate, double discount_factor) :
	gd_updater(learning_rate),
	discount_factor_(discount_factor) {}

rmspropupdater::~rmspropupdater (void)
{
	for (variable<double>* momentum : momentums_)
	{
		delete momentum;
	}
}

rmspropupdater* rmspropupdater::clone (void) { return static_cast<rmspropupdater*>(clone_impl()); }

rmspropupdater* rmspropupdater::move (void) { return static_cast<rmspropupdater*>(move_impl()); }

gd_updater* rmspropupdater::clone_impl (void) const
{
	return new rmspropupdater(*this);
}

gd_updater* rmspropupdater::move_impl (void)
{
	return new rmspropupdater(std::move(*this));
}

rmspropupdater& rmspropupdater::operator = (const rmspropupdater& other)
{
	if (this != &other)
	{
		gd_updater::operator = (other);
		discount_factor_ = other.discount_factor_;
	}
	return *this;
}

rmspropupdater& rmspropupdater::operator = (rmspropupdater&& other)
{
	if (this != &other)
	{
		gd_updater::operator = (std::move(other));
		discount_factor_ = std::move(other.discount_factor_);
	}
	return *this;
}

void rmspropupdater::set_discount_factor (double discount_factor)
{
	discount_factor_ = discount_factor;
}

rmspropupdater::rmspropupdater (const rmspropupdater& other) :
	gd_updater(other),
	discount_factor_(other.discount_factor_) {}

rmspropupdater::rmspropupdater (rmspropupdater&& other) :
	gd_updater(std::move(other)),
	discount_factor_(std::move(other.discount_factor_)) {}
	
variable_updater<double> rmspropupdater::process_update (varptr<double>& gres,
	variable<double>* leaf, grad_process intermediate_process)
{
	const_init<double> wuninit(1);
	variable<double>* momentum = new variable<double>(leaf->get_shape(), wuninit, "momentum");
	momentum->initialize();
	momentums_.push_back(momentum); // updater manages momentum variable

	// momentum = discount_factor_ * momentum + (1 - discount_factor_) * gres^2
	// leaf = leaf - learning_rate * gres / (sqrt(momentum) + epsilon)
	varptr<double> dres = intermediate_process(gres, leaf);
	varptr<double> momentum_step = discount_factor_ * varptr<double>(momentum) + (1-discount_factor_) * pow(dres, 2);
	varptr<double> leaf_step = dres * learning_rate_ / (sqrt<double>(momentum_step) + epsilon_);
	auto momentum_update = momentum->assign(momentum_step);
	auto leaf_update = leaf->assign_sub(leaf_step);
	return [momentum_update, leaf_update, momentum, dres, leaf_step](bool notify)
	{
		momentum_update(false);
		leaf_update(notify);
	};
}


}

#endif