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

// updates position on error manifold
EVOKER_PTR<double> gd_optimizer::apply_grad (GRAD_MAP<double>& gradients) {
	GRAD_MAP<double> local_grad;
	std::shared_ptr<group<double> > g_ptr = std::make_shared<group<double> >();

	for (auto& g : gradients) {
		VAR_PTR<double> old_var = g.first;
		VAR_PTR<double> delta = g.second;
		// calculate all delta BEFORE updating
		g_ptr->add(std::static_pointer_cast<ievoker<double> >(delta));
		local_grad[old_var] = std::make_shared<var_buffer<double> >(delta);
	}

	for (auto& g : local_grad) {
		VAR_PTR<double> old_var = g.first;
		VAR_PTR<double> delta = g.second;
		EVOKER_PTR<double> evok = std::make_shared<update_sub<double> >(
			std::static_pointer_cast<variable<double>, ivariable<double> >(old_var), this->learning_rate_ * delta);
		g_ptr->add(evok);
	}

	return g_ptr;
}

// MOMENTUM BASED OPTIMIZATION
// updates velocity of positional update on error manifold

EVOKER_PTR<double> ada_delta_optimizer::apply_grad (GRAD_MAP<double>& gradients) {
	
	return nullptr;
}

EVOKER_PTR<double> ada_grad_optimizer::apply_grad (GRAD_MAP<double>& gradients) {
	
	return nullptr;
}

EVOKER_PTR<double> rms_prop_optimizer::apply_grad (GRAD_MAP<double>& gradients) {
	// declare order update here
	// TODO: make ordered update group (first then...)
	GRAD_MAP<double> intermediates;

	for (auto it = gradients.begin(); gradients.end() != it; it++) {
		VAR_PTR<double> old_var = (*it).first;
		VAR_PTR<double> rms_delta = (*it).second;

		// additional optimization?

		// TODO: determine initial value?
		VAR_PTR<double> rms = variable<double>::make(1);
		intermediates[old_var] = rms_delta / rms;

		EVOKER_PTR<double> evok = std::make_shared<update<double> >(
			std::static_pointer_cast<variable<double> >(rms), rms_delta,
			[this](double& out, double in) {
				out = (1-discount_factor_) * out + discount_factor_ * in * in;
			});
		// insert evok to order update
	}

	EVOKER_PTR<double> wb_update = gd_optimizer::apply_grad(intermediates);

	// wrap evok then wb_update

	return nullptr;
}

}

#endif