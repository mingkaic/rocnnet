//
// Created by Mingkai Chen on 2017-04-27.
//

#include "executor/gd_utils.hpp"

#ifdef ROCNNET_GD_UTILS_HPP

namespace nnet
{

std::vector<nnet::variable_updater<double> > bgd_utils::calculate (inode<double>* root)
{
	std::vector<nnet::variable_updater<double> > updates;
	nnet::inode<double>::GRAD_CACHE leafset;
	root->get_leaves(leafset);
	for (auto lit : leafset)
	{
		nnet::variable<double>* Wb = lit.first;
		nnet::varptr<double> gres = root->get_gradient(Wb);
		// Wb = Wb + learning_rate * gres
		updates.push_back(Wb->assign_add(gres * learning_rate_));
	}
	return updates;
}

}

#endif