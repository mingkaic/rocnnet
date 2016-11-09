//
//  extend.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef extend_hpp
#define extend_hpp

#include "graph/operation/unary/iunar_ops.hpp"

namespace nnet {

// TENSOR EXTENSION

template<typename T>
class extend : public iunar_ops<T> {
	private:
		size_t index = 0;
		size_t multiplier = 0;
		WEAK_VAR_PTR<T> watch;

	protected:
		virtual void make_gradient(VAR_PTR<T> &safety_ref);

		virtual std::string get_symb(void) { return "extend"; }

		virtual void shape_eval(void);

		void copy(const ivariable<T> &other, std::string name = "");

		extend(const ivariable<T> &other, std::string name) { this->copy(other, name); }

		extend(VAR_PTR<T> in, WEAK_VAR_PTR<T> watch); // extend to fit shape
		extend(VAR_PTR<T> in, size_t index, size_t multiplier);

		virtual EVOKER_PTR<T> clone_impl(std::string name);

	public:
		static VAR_PTR<T> make(VAR_PTR<T> in, WEAK_VAR_PTR<T> watch) {
			VAR_PTR<T> root = ivariable<T>::make_shared(new extend(in, watch));
			// TODO: come up with a dryer solution to handling inherited attribute nodes (perhaps treat every node as inherited?)
			// have each argument evaluate interaction root
			in->interact(root);
			return root;
		}

		static VAR_PTR<T> make(VAR_PTR<T> in, size_t index = 0, size_t multiplier = 1) {
			VAR_PTR<T> root = ivariable<T>::make_shared(new extend(in, index, multiplier));
			// TODO: come up with a dryer solution to handling inherited attribute nodes (perhaps treat every node as inherited?)
			// have each argument evaluate interaction root
			in->interact(root);
			return root;
		}

		virtual extend<T> &operator=(const ivariable<T> &other);

		std::shared_ptr<extend<T> > clone(std::string name = "") {
			return std::static_pointer_cast<extend<T>, ievoker<T> >(clone_impl(name));
		}

		// set data
		void set_ext_info(WEAK_VAR_PTR<T> watch);

		void set_ext_info(size_t index, size_t multiplier);

		virtual const tensor<T> &eval(void);
};

}

#include "../../../../../src/graph/operation/unary/matrix_op/extend.ipp"

#endif /* extend_hpp */
