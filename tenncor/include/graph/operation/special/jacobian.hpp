//
//  jacobian.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-06.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef jacobian_hpp
#define jacobian_hpp

#include "graph/operation/ioperation.hpp"

namespace nnet {

// higher-dimensional unaligned operational tensor

// jacobian is really a tensor of tensors
// (each tensor being an abstraction of either raw data, or some system of equations)
// currently, tensor of tensor operation isn't well defined, so instead only accept jacobian vectors,
// vectors of tensors gradients (more specifically, gradients of matrix multiplication)

// the jacobian is a two part node, the publicly exposed version which always returns one as the evaluation
// and the private version which returns the jacobian tensor (tensor_jacobi)
// the jacobian tensor must be manually connected to other nodes via its set_root method

template <typename T>
class jacobian : public ioperation<T> {
	private:
		class hidden_jacobi : public ioperation<T> {
			protected:
				virtual void setup_gradient (void) {}
				// this isn't ever used... tensor_jacobi evaluates its own shape
				virtual tensorshape shape_eval (void) { return std::vector<size_t>{}; }

				virtual ivariable<T>* clone_impl (std::string name) {
					return new hidden_jacobi(*this, name);
				}

				hidden_jacobi (const hidden_jacobi& other, std::string name) :
						ioperation<T>(other, name) {}

			public:
				hidden_jacobi (jacobian<T>* outer, bool transposeA, bool transposeB) :
						ioperation<T>(std::vector<ivariable<T>*>{outer}, "jacobian_hidden") {
					this->out_ = std::make_unique<tensor_jacobi<T> >(transposeA, transposeB);
				}

				hidden_jacobi* clone (std::string name = "") {
					return static_cast<hidden_jacobi*>(clone_impl(name));
				}

				hidden_jacobi& operator () (ivariable<T>* a, ivariable<T>* b) { (*this->out_)(a, b); }

				virtual void update (ccoms::subject* caller) {
					this->notify();
				}
		};

		hidden_jacobi hidden_;

	protected:
		virtual void setup_gradient (void) {
			// jacobian is inherently a gradient agent (it exists in some first order or above derivative function)
			// TODO will do nothing until second/nth order derivative is implemented
		}

		virtual tensorshape shape_eval (void) { return std::vector<size_t>{}; }

		virtual ivariable<T>* clone_impl (std::string name) {
			return new jacobian(*this, name);
		}

		virtual bool channel (std::stack<ivariable<T>*>& jacobi) {
			jacobi.push(&hidden_);
			// ioperation's channel looks in dependencies indiscriminately,
			// which could be branching to some gradient state node (TODO: optimize once gradient order is implemented)
			// however, in jacobian's case, its dependencies are guaranteed non-gradient nodes,
			// so don't bother using ioperation<T>::jacobi to propagate channel
			size_t jacobi_count = 0;
			for (ccoms::subject* sub : this->dependencies_) {
				if (ioperation<T>* o = dynamic_cast<ioperation<T>*>(sub)) {
					// operation node's gradient SHOULD be an operation
					ioperation<T>* go = static_cast<ioperation<T>*>(o->get_gradient());
					if (go->channel(jacobi)) {
						jacobi_count++;
					}
				}
			}
			if (jacobi_count > 1) {
				throw std::logic_error("jacobian branch conflict occurred at " + this->get_name());
			}
			return jacobi_count != 0;
		}

		jacobian (const jacobian<T>& other, std::string name) :
			ioperation<T>(other, name),
			hidden_(other.hidden_) {}

	public:
		jacobian (ivariable<T>* arga, ivariable<T>* argb,
				bool transposeA, bool transposeB,
				std::string name = "") :
			ioperation<T>(std::vector<ivariable<T>*>{arga, argb}, name),
			hidden_(transposeA, transposeB) { this->out_ = std::make_unique<tensor<T> >(1); }

		jacobian<T>* clone (std::string name = "") {
			return static_cast<jacobian<T>*>(clone_impl(name));
		}

		virtual void update (ccoms::subject* caller) {
			ivariable<T>* a = dynamic_cast<ivariable<T>*>(this->dependencies_[0]);
			ivariable<T>* b = dynamic_cast<ivariable<T>*>(this->dependencies_[1]);
			if (a && b) {
				hidden_(a, b);
			}
			this->notify();
		}
};

}

#include "../../../../src/graph/operation/special/jacobian.ipp"

#endif /* jacobian_hpp */
