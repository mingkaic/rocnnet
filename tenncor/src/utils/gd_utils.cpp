//
// Created by Mingkai Chen on 2017-04-27.
//

#include "utils/gd_utils.hpp"

#ifdef ROCNNET_GD_UTILS_HPP

namespace nnet
{


vgb_updater*  vgb_updater::clone (void) { return static_cast<vgb_updater*>(clone_impl()); }

vgb_updater*  vgb_updater::move (void) { return static_cast<vgb_updater*>(move_impl()); }

gd_updater<double>*  vgb_updater::clone_impl (void)
{
	return new vgb_updater(*this);
}

gd_updater<double>* vgb_updater::move_impl (void)
{
	return new vgb_updater(std::move(*this));
}

nnet::variable_updater<double> vgb_updater::process_update (varptr<double>& gres,
	variable<double>* leaf, grad_process<double> intermediate_process)
{
	// leaf = leaf - learning_rate * gres
	return leaf->assign_sub(intermediate_process(gres, leaf) * learning_rate_);
}


momentum_updater* momentum_updater::clone (void) { return static_cast<momentum_updater*>(clone_impl()); }

momentum_updater* momentum_updater::move (void) { return static_cast<momentum_updater*>(move_impl()); }

gd_updater<double>* momentum_updater::clone_impl (void)
{
	return new momentum_updater(*this);
}

gd_updater<double>* momentum_updater::move_impl (void)
{
	return new momentum_updater(std::move(*this));
}

variable_updater<double> momentum_updater::process_update (varptr<double>& /*gres*/,
	variable<double>* /*leaf*/, grad_process<double> /*intermediate_process*/)
{
	throw std::bad_function_call();
	return [](void) {};
}


adadelta_updater* adadelta_updater::clone (void) { return static_cast<adadelta_updater*>(clone_impl()); }

adadelta_updater* adadelta_updater::move (void) { return static_cast<adadelta_updater*>(move_impl()); }

gd_updater<double>* adadelta_updater::clone_impl (void)
{
	return new adadelta_updater(*this);
}

gd_updater<double>* adadelta_updater::move_impl (void)
{
	return new adadelta_updater(std::move(*this));
}

variable_updater<double> adadelta_updater::process_update (varptr<double>& /*gres*/,
	variable<double>* /*leaf*/, grad_process<double> /*intermediate_process*/)
{
	throw std::bad_function_call();
	return [](void) {};
}


adagradupdater* adagradupdater::clone (void) { return static_cast<adagradupdater*>(clone_impl()); }

adagradupdater* adagradupdater::move (void) { return static_cast<adagradupdater*>(move_impl()); }

gd_updater<double>* adagradupdater::clone_impl (void)
{
	return new adagradupdater(*this);
}

gd_updater<double>* adagradupdater::move_impl (void)
{
	return new adagradupdater(std::move(*this));
}

variable_updater<double> adagradupdater::process_update (varptr<double>& /*gres*/,
	variable<double>* /*leaf*/, grad_process<double> /*intermediate_process*/)
{
	throw std::bad_function_call();
	return [](void) {};
}


//rmspropupdater::rmspropupdater (void) :
//	momentum_() {}

rmspropupdater* rmspropupdater::clone (void) { return static_cast<rmspropupdater*>(clone_impl()); }

rmspropupdater* rmspropupdater::move (void) { return static_cast<rmspropupdater*>(move_impl()); }

gd_updater<double>* rmspropupdater::clone_impl (void)
{
	return new rmspropupdater(*this);
}

gd_updater<double>* rmspropupdater::move_impl (void)
{
	return new rmspropupdater(std::move(*this));
}

variable_updater<double> rmspropupdater::process_update (varptr<double>& gres,
	variable<double>* leaf, grad_process<double> intermediate_process)
{
	// momentum = discount_factor_ * momentum + (1 - discount_factor_) * gres^2
	// leaf = leaf - learning_rate * gres / sqrt(momentum) + epsilon
//	varptr<double> dres = intermediate_process(gres, leaf);
//	varptr<double> momentumstep = ;
//	varptr<double> leafstep = - dres * learning_rate_ / sqrt<double>(momentum_) + epsilon_;
//
//	return leaf->assign_add(leafstep);
}


}

#endif