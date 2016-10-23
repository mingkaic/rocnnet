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
	return std::vector<GRAD<double> >();
}

// update variables not covered by ignore
void gd_optimizer::apply_grad (std::vector<GRAD<double> > gradients) {

}

// separate minimize into the steps
std::vector<GRAD<double> > rms_prop_optimizer::compute_grad (VAR_PTR<double> fanout) {
	return std::vector<GRAD<double> >();
}

// update variables not covered by ignore
void rms_prop_optimizer::apply_grad (std::vector<GRAD<double> > gradients) {

}

}

#endif