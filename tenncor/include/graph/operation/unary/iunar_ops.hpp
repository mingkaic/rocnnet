//
//  iunar_ops.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-09-30.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef unar_ops_hpp
#define unar_ops_hpp

#include "graph/operation/ioperation.hpp"

namespace nnet {

// UNARY OPERATIONS

template <typename T>
class iunar_ops : public ioperation<T> {
	protected:
		// setup_gradient remains abstract
		virtual void shape_eval (void);
		// clone_impl remains abstract
		
		virtual std::string get_symb (void) = 0;

		friend class univar_func<T>;

	public:
		iunar_ops (ivariable<T>* arg)
				: ioperation<T>(std::vector<ivariable<T>*>{arg},
				nnutils::formatter() << "<" << get_symb() << ">(" << arg->get_name() << ")") {
			if (session::pre_shape_eval()) {
				shape_eval();
			}
		}
};

}

#include "../../../../src/graph/operation/unary/iunar_ops.ipp"

#endif /* unar_ops_hpp */
