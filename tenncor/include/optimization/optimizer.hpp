//
//  optimizer.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-22.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <vector>
#include <map>

#pragma once
#ifndef optimizer_hpp
#define optimizer_hpp

#include "update.hpp"
#include "graph/variable/variable.hpp"
#include "graph/group.hpp"
#include "graph/operation/elementary.hpp"
#include "graph/operation/unary/iunar_ops.hpp"
#include "graph/operation/unary/derive.hpp"
#include "graph/observer/node.hpp"

namespace nnet {

template <typename T>
using GRAD_MAP = std::map<ivar_init<T>*, VAR_PTR<T> >;

// optimizers compute and update the variables (weights and biases) 
// by first computing the gradients then updating them 
// via their respective update algorithm
// this process is broken into 2 parts in order to give the caller 
// the option to customize gradients prior update.
// often, mini-batch normalization occurs between 
// gradient calculation and update

template <typename T>
class ioptimizer {
	protected:
		std::unordered_set<ccoms::inode*> ignore_set_;

	public:
		virtual ~ioptimizer (void) {}

		void ignore (VAR_PTR<T> ig_var) { ignore_set_.insert(ig_var.get()); }

		// two step process in one
		EVOKER_PTR<T> minimize (VAR_PTR<T> fanout) {
			GRAD_MAP<T> buffer = this->compute_grad(fanout);
			return apply_grad(buffer);
		}

		// separate minimize into 2 steps:
		// calculate the gradient
		virtual GRAD_MAP<T> compute_grad (VAR_PTR<T> fanout) {
			GRAD_MAP<T> res;
			fanout->leaves_collect ([&res, this](ccoms::subject* leaf) {
				if (ignore_set_.end() == ignore_set_.find(leaf)) {
					std::pair<VAR_PTR<T>, VAR_PTR<T> > end_point(
						dynamic_cast<ivar_init<T>*>(leaf),
						derive<T>::make(fanout, leaf));
					res.insert(end_point);
				}
			});
			return res;
		}
		// actual update step
		virtual EVOKER_PTR<T> apply_grad (GRAD_MAP<T>& gradients) = 0;
};

template <typename T>
using OPTIMIZER = std::shared_ptr<ioptimizer<T> >;

// Gradient Descent Update Algorithm

// updates position on error manifold
// assume that input operation is some cost function J
// input gradient of J gives derivative J'[v] (wrt v) for each variable v
// update by gradient descent algorithm:
// var_t = var_(t-1) - delta(var)
// where delta(var) = learning * J'[var_(t-1)]

class gd_optimizer : public ioptimizer<double> {
	private:
		double learning_rate_;

	public:
		gd_optimizer (double learning_rate) : learning_rate_(learning_rate) {}

		// inherits compute_grad from ioptimizer

		virtual EVOKER_PTR<double> apply_grad (GRAD_MAP<double>& gradients);
};

// MOMENTUM BASED OPTIMIZATION
// overview: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
// updates velocity of positional update on error manifold

// Standard momentum:
// 1. velocity_t = discount * velocity_(t-1) - learning * J'[v]
// 2. delta(var) = velocity_t, update by gd
// lim t->inf (velocity) = -learning * J'[v] / (1-discount)

// Nestrov momentum:
// 1. delta(var) = velocity_t-1, update by gd
// 2. velocity_t = discount * velocity_(t-1) - learning * J'[v]

// Separate adaptive learning rates
// introduce variable local_gain linked to weight/bias variables
// delta(var) = -epsilon * local_gain[v] * J'[v]
// if J'[v]_t * J'[v]_(t-1) > 0:
// then local_gain[v] += 0.05
// else local_gain[v] *= 0.95

class ada_delta_optimizer : public gd_optimizer {
	private:
		double rho_;
		double epsilon_;

	public:
		ada_delta_optimizer (double learning_rate,
							double rho = 0.95,
							double epsilon = std::numeric_limits<double>::epsilon()) :
			gd_optimizer(learning_rate), rho_(rho), epsilon_(epsilon) {}

		// inherits compute_grad from ioptimizer

		virtual EVOKER_PTR<double> apply_grad (GRAD_MAP<double>& gradients);
};

class ada_grad_optimizer : public gd_optimizer {
	private:
		double init_accum_;

	public:
		ada_grad_optimizer (double learning_rate,
							double init_accum = 0.1) :
			gd_optimizer(learning_rate), init_accum_(init_accum) {}

		// inherits compute_grad from ioptimizer

		virtual EVOKER_PTR<double> apply_grad (GRAD_MAP<double>& gradients);
};

// Root Mean Square Propagation Algorithm
// rms_delta = J'(v)_t
// rms_t = (1 - discount) * rms_t-1 + discount * rms_delta^2
// delta(var) = v_t = learning * rms_delta / rms_t

// there maybe momentum implementation to...
// change to rms_delta

class rms_prop_optimizer : public gd_optimizer {
	private:
		// input variables
		double discount_factor_;
		double momentum_;
		double epsilon_; // for adaptive learning rates

	public:
		rms_prop_optimizer (double learning_rate,
							double discount_factor = 0.9,
							double momentum = 0.0,
							double epsilon = std::numeric_limits<double>::epsilon()) :
			gd_optimizer(learning_rate),
			discount_factor_(discount_factor),
			momentum_(momentum),
			epsilon_(epsilon) {}

		// inherits compute_grad from ioptimizer

		virtual EVOKER_PTR<double> apply_grad (GRAD_MAP<double>& gradients);
};

}

#endif /* optimizer_hpp */
