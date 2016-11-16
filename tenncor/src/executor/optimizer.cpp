//
//  optimizer.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-22.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "executor/optimizer.hpp"

#ifdef optimizer_hpp

namespace nnet {
	
gd_optimizer::gd_optimizer (double learning_rate) : 
	learning_rate_(learning_rate) {}

// updates position on error manifold
group<double>* gd_optimizer::apply_grad (void) const
{
	group<double>* g_ptr = new group<double>();

	for (auto& top_pair : grad_top_)
	{
		if (variable<double>* old_var = 
			dynamic_cast<variable<double>*>(top_pair.first))
		{
			ivariable<double> *delta = top_pair.second;
			// pass assign_sub ownership to g_ptr group
			g_ptr->add(new assign_sub<double>(old_var, delta), true);
		}
	}
	return g_ptr;
}

// MOMENTUM BASED OPTIMIZATION
// updates velocity of positional update on error manifold

group<double>* ada_delta_optimizer::apply_grad (void) const
{
	
	return nullptr;
}

group<double>* ada_grad_optimizer::apply_grad (void) const
{
	
	return nullptr;
}

rms_prop_optimizer::rms_prop_optimizer (
	double learning_rate, double discount_factor,
	double momentum,
	double epsilon) :
	gd_optimizer(learning_rate),
	discount_factor_(discount_factor),
	momentum_(momentum),
	epsilon_(epsilon) {}

group<double>* rms_prop_optimizer::apply_grad (void) const
{
	// declare order update here
	// TODO: rms prop WIP
	// GRAD_MAP<double> intermediates;

	// for (auto it = gradients.begin(); gradients.end() != it; it++)
	// {
	// 	nnet::varptr<double> old_var = (*it).first;
	// 	nnet::varptr<double> rms_delta = (*it).second;

	// 	// do something to rms_delta... before plugging into intermedates/assignment

	// 	intermediates[old_var] = rms_delta;

	// 	// add additional buffering...
	// }

	// group<double>* wb_update = gd_optimizer::apply_grad(intermediates);
	return nullptr;
}

}

#endif