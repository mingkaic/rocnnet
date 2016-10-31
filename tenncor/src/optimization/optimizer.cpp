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

// update variables not covered by ignore
EVOKER_PTR<double> gd_optimizer::apply_grad (GRAD_MAP<double>& gradients) {
	// assume that input fanout is some cost function J
	// gradient of J[v] gives delta J[v] for each variable v
	// including placeholders. (ignore placeholders and update variables only)

	// update by equation var_new = var_old - learning * delta(J(var_old))
	// for each weight/bias var in graph

	for (auto it = gradients.begin(); gradients.end() != it; it++) {
		VAR_PTR<double> old_var = (*it).first;
		VAR_PTR<double> delta = (*it).second;

		EVOKER_PTR<double> evok = std::make_shared<update_sub<double> >(
			std::static_pointer_cast<variable<double>, ivariable<double> >(old_var), delta);
	}

	// wrap in group evoker

	return nullptr;
}

// update variables not covered by ignore
EVOKER_PTR<double> rms_prop_optimizer::apply_grad (GRAD_MAP<double>& gradients) {
	return nullptr;
}

}

#endif