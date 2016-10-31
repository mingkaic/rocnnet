//
//  expose.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-09-30.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef expose_hpp
#define expose_hpp

#include "iunar_ops.hpp"

namespace nnet {

// OUT NODE

template <typename T>
class expose : public iunar_ops<T> {
	protected:
		// backward chaining for AD
		virtual void make_gradient (VAR_PTR<T>& safety_ref) {}

		std::vector<T> get_vec (const tensor<T>& in) const;

		expose (ivariable<T>& var, std::string name) { this->copy(var, name); }
		expose (VAR_PTR<T> var) { this->init(var); }

		std::string get_symb (void) { return "expose"; }
		virtual EVOKER_PTR<T> clone_impl (std::string name);

	public:
		static std::shared_ptr<expose<T> > make (VAR_PTR<T> var) {
			VAR_PTR<T> inst = ivariable<T>::make_shared(new expose(var));
			return std::static_pointer_cast<expose<T> >(inst);
		}

		virtual const tensor<T>& eval (void);

		std::shared_ptr<expose<T> > clone (std::string name = "") {
			return std::static_pointer_cast<expose<T>, ievoker<T> >(clone_impl(name));
		}

		// non-inheriteds
		// evaluates consumed operation
		virtual std::vector<T> get_raw (void);
		// extracts derivative based on the LAST evaluation
		// doesn't evaluate
		virtual std::vector<T> get_derive (VAR_PTR<T> over) const;
};

}

#include "../../../../src/graph/operation/unary/matrix_op/expose.ipp"

#endif /* expose_hpp */
