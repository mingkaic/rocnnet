//
//  optimizer.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-22.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <vector>
#include "../variable/variable.hpp"

#pragma once
#ifndef optimizer_hpp
#define optimizer_hpp

#include "update.hpp"

namespace nnet {

template <typename T>
using GRAD = std::pair<VAR_PTR<T>, VAR_PTR <T> >;

template <typename T>
class ioptimizer {
	protected:
		std::unordered_set<VAR_PTR<T> > ignore_set;

		double learning_rate;

	public:
		virtual ~ioptimizer (void) {}

		void ignore (VAR_PTR<T> ig_var) { ignore_set.emplace(ig_var); }

		// two step process in one
		EVOKER_PTR<T> minimize (VAR_PTR<T> fanout) {
			apply_grad(compute_grad(fanout));
		}

		// separate minimize into the steps
		virtual std::vector<GRAD<T> > compute_grad (VAR_PTR<T> fanout) = 0;
		// update variables not covered by ignore
		virtual EVOKER_PTR<T> apply_grad (std::vector<GRAD<T> > gradients) = 0;
};

template <typename T>
using OPTIMIZER = std::shared_ptr<ioptimizer<T> >;

// gradient descent
class gd_optimizer : public ioptimizer<double> {
	public:
		// separate minimize into the steps
		virtual std::vector<GRAD<double> > compute_grad (VAR_PTR<double> fanout);
		// update variables not covered by ignore
		virtual EVOKER_PTR<double> apply_grad (std::vector<GRAD<double> > gradients);
};

// rms prop
class rms_prop_optimizer : public ioptimizer<double> {
	private:
		double discount_factor;
		double momentum;
		double epsilon;

	public:
		rms_prop_optimizer (double learning_rate,
							double discount_factor = 0.9,
							double momentum = 0.0,
							double epsilon = std::numeric_limits<double>::epsilon()) :
			discount_factor(discount_factor), momentum(momentum), epsilon(epsilon) {
			this->learning_rate = learning_rate;
		}

		// separate minimize into the steps
		virtual std::vector<GRAD<double> > compute_grad (VAR_PTR<double> fanout);
		// update variables not covered by ignore
		virtual EVOKER_PTR<double> apply_grad (std::vector<GRAD<double> > gradients);
};

}

#endif /* optimizer_hpp */
