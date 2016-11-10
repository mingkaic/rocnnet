//
//  derive.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef derive_hpp
#define derive_hpp

#include "iunar_ops.hpp"

namespace nnet {

// DERIVE

// TODO test derivation
template <typename T>
class derive : public iunar_ops<T> {
	private:
		ivariable<T>* over_ = nullptr;

		derive (ivariable<T>* var, std::string name) { this->copy(var, name); } // copy constructor

	protected:
		std::string get_symb (void) { return "/derive(" + over_->get_name() + ")?"; }

		virtual ievoker<T>* clone_impl (std::string name);
		virtual void setup_gradient (void) {
			// TODO implement second order calc_gradient
		}

	public:
		derive (ivariable<T>* func, ivariable<T>* over) :
			iunar_ops<T>(func->get_derive()), over_(over) {}

		virtual const tensor<T>& get_eval (void) {
			if (this->short_circuit) {
				return this->ones;
			}
			if (ioperation<T>* func = dynamic_cast<ioperation<T>*>(this->dependencies_[0])) {
				func->leaves_collect([this](ccoms::subject* sub){
					if (over_ != sub) {
						if (ivariable<T> *leaf = dynamic_cast<ivariable<T> *>(sub)) {
							leaf->short_circuit = false;
						}
					}
				});
			}
			over_->notify();
			return this->out_;
		}

		virtual void update (void);
        derive<T>* clone (std::string name = "") {
			return static_cast<derive<T>*>(clone_impl(name));
		}
};

}

#include "../../../../src/graph/operation/unary/derive.ipp"

#endif /* derive_hpp */
