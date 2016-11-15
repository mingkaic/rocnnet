//
//  optimizer.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-22.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/executor/optimizer.hpp"

#ifdef optimizer_hpp

namespace nnet {
	
gd_optimizer::gd_optimizer (double learning_rate) : 
	learning_rate_(learning_rate) {}

// updates position on error manifold
void gd_optimizer::execute (void)
{
	GRAD_MAP<double> local_grad;
	group<double>* g_ptr = new group<double>();

	for (auto& g : gradients)
	{
		if (variable<double>* old_var = 
			dynamic_cast<variable<double>*>(g.first))
		{
			ivariable<double> *delta = g.second;
			g_ptr->add(new assign_sub<double>(old_var, delta));
		}
	}

	return g_ptr;
}

// MOMENTUM BASED OPTIMIZATION
// updates velocity of positional update on error manifold

void ada_delta_optimizer::execute (void)
{
	
	return nullptr;
}

void ada_grad_optimizer::execute (void)
{
	
	return nullptr;
}

rms_prop_optimizer::rms_prop_optimizer (
	double learning_rate, double discount_factor = 0.9,
	double momentum = 0.0, 
	double epsilon = std::numeric_limits<double>::epsilon()) :
	gd_optimizer(learning_rate),
	discount_factor_(discount_factor),
	momentum_(momentum),
	epsilon_(epsilon) {}

void rms_prop_optimizer::execute (void)
{
	// declare order update here
	// TODO: rms prop WIP
	GRAD_MAP<double> intermediates;

	for (auto it = gradients.begin(); gradients.end() != it; it++)
	{
		nnet::varptr<double> old_var = (*it).first;
		nnet::varptr<double> rms_delta = (*it).second;

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