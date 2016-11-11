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
		ivariable<T>* watch_ = nullptr;

		extend(const extend<T> &other, std::string name) { copy(other, name); }

	protected:
		virtual void setup_gradient(void);
		virtual ievoker<T>*clone_impl(std::string name);
		virtual void shape_eval(void);
		
		virtual std::string get_symb(void) { return "extend"; }

        void copy(const extend<T> &other, std::string name = "");

	public:
        extend (ivariable<T>* in, ivariable<T>* watch); // extend to fit shape
        extend (ivariable<T>* in, size_t index, size_t multiplier);

		// COPY
		extend<T>* clone(std::string name = "") {
			return static_cast<extend<T>*>(clone_impl(name));
		}
		virtual extend<T>& operator = (const ivariable<T> &other);

		virtual void update (void);

		// set data
		void set_ext_info(ivariable<T>* watch);

		void set_ext_info(size_t index, size_t multiplier);
};

}

#include "../../../../../src/graph/operation/unary/matrix_op/extend.ipp"

#endif /* extend_hpp */
