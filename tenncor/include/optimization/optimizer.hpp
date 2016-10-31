//
//  optimizer.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-22.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <vector>
#include <map>
#include "graph/variable/variable.hpp"
#include "graph/operation/unary/iunar_ops.hpp"

#pragma once
#ifndef optimizer_hpp
#define optimizer_hpp

#include "update.hpp"
#include "../graph/group.hpp"
#include "../graph/operation/elementary.hpp"
#include "../graph/operation/unary/derive.hpp"

namespace nnet {

template <typename T>
using GRAD_MAP = std::map<VAR_PTR<T>, VAR_PTR<T> >;

template <typename T>
class ioptimizer {
	protected:
		nnutils::WEAK_SET<ivariable<T> > ignore_set;
		double learning_rate;

	public:
		virtual ~ioptimizer (void) {}

		void ignore (VAR_PTR<T> ig_var) { ignore_set.insert(ig_var); }

		// two step process in one
		EVOKER_PTR<T> minimize (VAR_PTR<T> fanout) {
			apply_grad(compute_grad(fanout));
		}

		// separate minimize into the steps
		virtual GRAD_MAP<T> compute_grad (VAR_PTR<T> fanout) {
			GRAD_MAP<T> res;
			nnutils::WEAK_SET<ivariable<T> >& leaves = fanout->_leaves;

			for (WEAK_VAR_PTR<T> leaf : leaves) {
				if (ignore_set.end() == ignore_set.find(leaf)) {
					std::pair<VAR_PTR<T>, VAR_PTR<T> > end_point(leaf.lock(), derive<T>::make(fanout, leaf.lock()));
					res.insert(end_point);
				}
			}
			return res;
		}
		// update variables not covered by ignore
		virtual EVOKER_PTR<T> apply_grad (GRAD_MAP<T>& gradients) = 0;
};

template <typename T>
using OPTIMIZER = std::shared_ptr<ioptimizer<T> >;

// gradient descent
class gd_optimizer : public ioptimizer<double> {
	public:
	gd_optimizer (double learning_rate) {
			this->learning_rate = learning_rate;
		}

		// inherits compute_grad from ioptimizer

		// update variables not covered by ignore
		virtual EVOKER_PTR<double> apply_grad (GRAD_MAP<double>& gradients);
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

		// inherits compute_grad from ioptimizer

		// update variables not covered by ignore
		virtual EVOKER_PTR<double> apply_grad (GRAD_MAP<double>& gradients);
};

}

#endif /* optimizer_hpp */
