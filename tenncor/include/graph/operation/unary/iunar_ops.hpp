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
		virtual void shape_eval (void);
		
		virtual std::string get_symb (void) = 0;

		friend class univar_func<T>;

	public:
		iunar_ops (ivariable<T>* arg)
				: ioperation (std::vector<subject*>{arg}, 
				nnutils::formatter() << "<" << get_symb() << ">(" << arg->get_name() << ")") {
			if (session::pre_shape_eval()) {
				shape_eval();
			}
		}
};

// USED FOR ELEMENT WISE OPERATIONS ONLY

template <typename T>
class iunar_elem_ops : public iunar_ops<T> {
	protected:
		virtual std::function<T(T)> get_op (void) = 0; // these are for elementary and simple operations

	public:
		virtual void update (void);
};

}

#include "../../../../src/graph/operation/unary/iunar_ops.ipp"

#endif /* unar_ops_hpp */
