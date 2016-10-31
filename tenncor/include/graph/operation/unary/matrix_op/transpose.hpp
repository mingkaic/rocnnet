//
// Created by Mingkai Chen on 2016-10-30.
//

#pragma once
#ifndef transpose_hpp
#define transpose_hpp

#include "graph/operation/unary/iunar_ops.hpp"

namespace nnet {

// MATRIX TRANSPOSE

template <typename T>
class transpose : public iunar_ops<T> {
	protected:
		// backward chaining for AD
		virtual void make_gradient (VAR_PTR<T>& safety_ref);
		virtual std::string get_symb (void) { return "transpose"; }

		virtual void shape_eval (void);
		transpose (const ivariable<T>& other, std::string name) { this->copy(other, name); }
		transpose (VAR_PTR<T> in);

		virtual EVOKER_PTR<T> clone_impl (std::string name);

	public:
		static VAR_PTR<T> make (VAR_PTR<T> in) {
			return ivariable<T>::make_shared(new transpose(in));
		}

		std::shared_ptr<transpose<T> > clone (std::string name = "") {
			return std::static_pointer_cast<transpose<T>, ievoker<T> >(clone_impl(name));
		}

		virtual const tensor<T>& eval (void);
};

}

#include "../../../../../src/graph/operation/unary/matrix_op/transpose.ipp"

#endif /* transpose_hpp */
