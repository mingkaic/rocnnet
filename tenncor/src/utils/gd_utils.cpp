//
// Created by Mingkai Chen on 2017-04-27.
//

#include "utils/gd_utils.hpp"

#ifdef ROCNNET_GD_UTILS_HPP

namespace nnet
{

nnet::variable_updater<double> vgb_updater::process_update (varptr<double>& gres,
	variable<double>* leaf, grad_process<double> intermediate_process)
{
	// leaf = leaf - learning_rate * gres
	return leaf->assign_sub(intermediate_process(gres, leaf) * learning_rate_);
}

}

#endif