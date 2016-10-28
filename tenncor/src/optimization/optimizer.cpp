//
//  optimizer.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-22.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "../../include/optimization/optimizer.hpp"

#ifdef optimizer_hpp

namespace nnet {

// separate minimize into the steps
std::vector<GRAD<double> > gd_optimizer::compute_grad (VAR_PTR<double> fanout) {
	VAR_PTR<double> grad_tree = fanout->get_gradient();

	// get list of gradients from grad_tree

	// assume that input fanout is some cost function J

	// gradient of J[v] gives delta J[v] for each variable v
	// including placeholders. (ignore placeholders and update variables only)

	return std::vector<GRAD<double> >();
}

// update variables not covered by ignore
EVOKER_PTR<double> gd_optimizer::apply_grad (std::vector<GRAD<double> > gradients) {
	// update by equation var_new = var_old - learning * delta(J(var_old))
	// for each weight/bias var in graph

	for (GRAD<double> g : gradients) {
		VAR_PTR<double> old_var = g.second;
		VAR_PTR<double> delta = g.first;

		EVOKER_PTR<double> evok = std::make_shared<update_sub<double> >(
			std::static_pointer_cast<variable<double>, ivariable<double> >(old_var), delta);
	}

	// wrap in group evoker

	return nullptr;
}

// separate minimize into the steps
std::vector<GRAD<double> > rms_prop_optimizer::compute_grad (VAR_PTR<double> fanout) {

	return std::vector<GRAD<double> >();
}

// update variables not covered by ignore
EVOKER_PTR<double> rms_prop_optimizer::apply_grad (std::vector<GRAD<double> > gradients) {
	return nullptr;
}

}

#endif