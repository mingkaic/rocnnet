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
	private:
		transpose (const transpose<T>& other, std::string name) { ioperation<T>::copy(other, name); }

	protected:
		// backward chaining for AD
		virtual void setup_gradient (void);
		virtual ievoker<T>* clone_impl (std::string name);
		virtual std::string get_symb (void) { return "transpose"; }

		virtual void shape_eval (void);

	public:
		transpose (ivariable<T>* in);

        transpose<T>* clone (std::string name = "") {
			return static_cast<transpose<T>*>(clone_impl(name));
		}

		virtual void update (void);
};

}

#include "../../../../../src/graph/operation/unary/matrix_op/transpose.ipp"

#endif /* transpose_hpp */
