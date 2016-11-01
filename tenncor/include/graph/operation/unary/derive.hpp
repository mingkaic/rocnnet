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
		VAR_PTR<T> over_ = nullptr;

	protected:
		derive (ivariable<T>& var, std::string name) { this->copy(var, name); } // copy constructor
		derive (VAR_PTR<T> func, VAR_PTR<T> over) {
			this->over_ = over;
			this->init(func);
		}

		std::string get_symb (void) { return "/derive(" + over_->get_name() + ")?"; }
		virtual EVOKER_PTR<T> clone_impl (std::string name);

		virtual void make_gradient (VAR_PTR<T>& safety_ref) {
			// TODO implement second order calc_gradient
		}

	public:
		static VAR_PTR<T> make (VAR_PTR<T> func, VAR_PTR<T> over) {
			return ivariable<T>::make_shared(new derive(func, over));
		}
		virtual const tensor<T>& eval (void);
		std::shared_ptr<derive<T> > clone (std::string name = "") {
			return std::static_pointer_cast<derive<T>, ievoker<T> >(clone_impl(name));
		}
};

}

#include "../../../../src/graph/operation/unary/derive.ipp"

#endif /* derive_hpp */
