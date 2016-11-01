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
	std::shared_ptr<group<double> > g_ptr = std::make_shared<group<double> >();

	for (auto it = gradients.begin(); gradients.end() != it; it++) {
		VAR_PTR<double> old_var = (*it).first;
		VAR_PTR<double> delta = (*it).second;

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
	for (auto it = gradients.begin(); gradients.end() != it; it++) {
		VAR_PTR<double> old_var = (*it).first;
		VAR_PTR<double> rms_delta = (*it).second;
		
		// additional optimization?
		
		//VAR_PTR<double> rms = variable<double>::make(rms_delta);
		// std::make_shared<update<double> >(rms, rms_delta, 
		//	[this](double& out, double in) { rms = (1-discount) * rms + discount * rms_delta * rms_delta; }
	}
	
	// plug into gd_optimizer::apply_gradient
	
	return nullptr;
}

}

#endif