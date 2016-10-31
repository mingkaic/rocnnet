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
		// avoid calling ivariable's assignment multiple time
		VAR_PTR<T> var = nullptr;

		void copy (const ivariable<T>& other, std::string name = "");
		virtual void replace (ivariable<T>* food, VAR_PTR<T> newfood);
		virtual void shape_eval (void);
		// operator () getters
		virtual std::string get_symb (void) = 0;

		void init (VAR_PTR<T> var);

		friend class univar_func<T>;

	public:
		virtual ~iunar_ops (void) {}

		virtual iunar_ops<T>& operator = (const ivariable<T>& other);

		std::shared_ptr<iunar_ops<T> > clone (std::string name = "") {
			return std::static_pointer_cast<iunar_ops<T>, ievoker<T> >(this->clone_impl(name));
		}
};

// USED FOR ELEMENT WISE OPERATIONS ONLY

template <typename T>
class iunar_elem_ops : public iunar_ops<T> {
	protected:
		virtual std::function<T(T)> get_op (void) = 0; // these are for elementary and simple operations

	public:
		virtual ~iunar_elem_ops (void) {}
		std::shared_ptr<iunar_elem_ops<T> > clone (std::string name = "") {
			return std::static_pointer_cast<iunar_elem_ops<T>, ievoker<T> >(this->clone_impl(name));
		}

		virtual const tensor<T>& eval (void);
};

}

#include "../../../../src/graph/operation/unary/iunar_ops.ipp"

#endif /* unar_ops_hpp */
