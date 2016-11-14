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
group<double>* gd_optimizer::apply_grad (GRAD_MAP<double>& gradients) {
	GRAD_MAP<double> local_grad;
	group<double>* g_ptr = new group<double>();

	for (auto& g : gradients) {
		if (variable<double>* old_var = dynamic_cast<variable<double>*>(g.first)) {
			ivariable<double> *delta = g.second;
			g_ptr->add(new assign_sub<double>(old_var, delta));
		}
	}

	return g_ptr;
}

// MOMENTUM BASED OPTIMIZATION
// updates velocity of positional update on error manifold

group<double>* ada_delta_optimizer::apply_grad (GRAD_MAP<double>& gradients) {
	
	return nullptr;
}

group<double>* ada_grad_optimizer::apply_grad (GRAD_MAP<double>& gradients) {
	
	return nullptr;
}

group<double>* rms_prop_optimizer::apply_grad (GRAD_MAP<double>& gradients) {
	// declare order update here
	// TODO: rms prop WIP
	GRAD_MAP<double> intermediates;

	for (auto it = gradients.begin(); gradients.end() != it; it++) {
		nnet::ivariable<double>* old_var = (*it).first;
		nnet::ivariable<double>* rms_delta = (*it).second;

		// do something to rms_delta... before plugging into intermedates/assignment

		intermediates[old_var] = rms_delta;

		// add additional buffering...
	}

	group<double>* wb_update = gd_optimizer::apply_grad(intermediates);

	// wrap evok then wb_update

	return nullptr;
}

}

#endif